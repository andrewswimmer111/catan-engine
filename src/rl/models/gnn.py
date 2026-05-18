"""GNN policy/value network for the graph observation encoding.

Architecture overview
---------------------

The input observation is the flat 1704-D vector produced by
:class:`GraphObservationEncoder`. ``forward`` unpacks it into five
per-element tensors (tiles, vertices, edges, players, globals), runs the
typed graph trunk, and emits ``ACTION_SPACE_SIZE`` policy logits plus a
scalar value.

Forward pipeline:

1. **Player stream.** Per-seat rows pass through a Linear projection and
   one Transformer self-attention block. The output is a per-seat
   embedding tensor ``(B, 4, h_p)`` aware of the full player table.
2. **Owner injection.** Each vertex's settlement/city owner one-hot and
   each edge's road owner one-hot are used to gather the matching player
   embedding (or zero, for unowned slots). The result concatenates onto
   the vertex/edge input features before the trunk projects them, so the
   message-passing layers see ownership-aware embeddings directly.
3. **Per-type input projection.** Three Linear layers project tile,
   vertex, and edge feature vectors to a shared hidden dim ``h``. The
   projected stacks concatenate into a single ``(B, 145, h)`` node tensor
   in canonical order (tiles, then vertices, then edges).
4. **GNN trunk.** ``arch.n_mp_layers`` GATv2 layers operate on the
   batched-graph view: the static :data:`GRAPH_STRUCTURE.edge_index` is
   tiled across the batch with per-sample offsets, and the per-edge type
   one-hot is fed as ``edge_attr`` so the attention layer can learn
   relation-specific message functions. Each layer has a residual +
   :class:`LayerNorm`.
5. **Heads.** Four spatial heads consume their corresponding node
   embeddings (road→edge nodes, settle/city→vertex nodes, robber→tile
   nodes). A steal head consumes the per-seat player embedding. A global
   MLP head consumes a pooled board + player + global vector and
   produces the remaining 46 logits. The value head shares the global
   trunk output.
6. **Assembly.** Each head's outputs scatter into the canonical
   ``(B, ACTION_SPACE_SIZE)`` slots described by
   :data:`rl.encoding.action_routing.ACTION_ROUTING`. Illegal entries
   are filled with :data:`MASK_FILL_VALUE` and the final logits are
   safe to pass straight to :class:`Categorical`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.action_routing import ACTION_ROUTING, ActionHead, N_GLOBAL_ACTIONS
from rl.encoding.graph_observation import (
    EDGE_BLOCK,
    EDGE_FEAT_DIM,
    EDGE_OFFSET,
    EDGE_ROAD_SLOT_OFFSET,
    GLOBAL_BLOCK,
    GLOBAL_FEAT_DIM,
    GLOBAL_OFFSET,
    GRAPH_OBS_SHAPE,
    GRAPH_STRUCTURE,
    N_EDGE_TYPES,
    N_EDGES,
    N_PLAYERS,
    N_TILES,
    N_VERTICES,
    PLAYER_BLOCK,
    PLAYER_FEAT_DIM,
    PLAYER_OFFSET,
    TILE_BLOCK,
    TILE_FEAT_DIM,
    TILE_OFFSET,
    VERTEX_BLOCK,
    VERTEX_CITY_SLOT_OFFSET,
    VERTEX_FEAT_DIM,
    VERTEX_OFFSET,
    VERTEX_SETTLE_SLOT_OFFSET,
)
from rl.models.mlp import MASK_FILL_VALUE, ModelOutput
from rl.models.nets import (
    POLICY_HEAD_GAIN,
    TRUNK_GAIN,
    VALUE_HEAD_GAIN,
    orthogonal_init,
)

__all__ = ["GNNArch", "GNNPolicyValue", "DEFAULT_GNN_ARCH", "ValueKind"]


ValueKind = Literal["scalar", "vector"]
"""Value-head shape selector.

* ``"scalar"`` — single value per state, shape ``(B,)``. The legacy PPO
  path (which trains the value head against a per-seat shaped return) only
  has a meaningful target for one seat at a time, so it expects this
  shape.
* ``"vector"`` — full per-seat value vector rotated to viewer-as-slot-0,
  shape ``(B, N_PLAYERS)``. Slot ``i`` is the predicted win-probability
  for the seat at turn-order offset ``i`` from the encoder's viewer. This
  is the AlphaZero-style head; one batched leaf-eval call yields the
  whole MCTS backup vector without needing N forward passes.
"""


@dataclass(frozen=True)
class GNNArch:
    """Architecture knobs for :class:`GNNPolicyValue`.

    Defaults land at ~1.4M parameters — in the same order of magnitude as
    the flat MLP baseline (~3M) so a head-to-head comparison isolates the
    encoder choice rather than capacity. GNNs are more parameter-efficient
    than wide flat MLPs because each weight participates in many node
    updates; matching exactly would over-parameterise the GNN.
    """

    node_hidden: int = 256
    """Per-node hidden dim through the GNN trunk."""

    player_hidden: int = 128
    """Per-player embedding dim out of the player self-attention stream."""

    n_mp_layers: int = 4
    """Number of GATv2 message-passing layers."""

    n_heads: int = 4
    """Attention heads in each GATv2Conv and in the player self-attention."""

    global_mlp_hidden: int = 512
    """Hidden width of the global / value MLP that sits on the pooled embedding."""

    value_kind: ValueKind = "vector"
    """Value-head shape: ``"scalar"`` (B,) for PPO, ``"vector"`` (B, N_PLAYERS) for AZ.

    Defaults to ``"vector"``: Phase 3 (AlphaZero) is the supported training
    path and benefits from a per-seat value head (single forward replaces
    the four-perspective batched eval in MCTS). PPO callers must pin
    ``"scalar"`` explicitly because their per-step value target only
    covers one seat at a time.
    """


DEFAULT_GNN_ARCH: Final[GNNArch] = GNNArch()


class GNNPolicyValue(nn.Module):
    """Typed graph policy/value network.

    Returns the same :class:`ModelOutput` shape as :class:`MLPPolicyValue`
    so the rest of the trainer doesn't need to know which encoder produced
    it. The model is seat-agnostic — the encoder rotates the viewer into
    seat 0 on every forward pass — so weights are reusable across seats.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        arch: GNNArch = DEFAULT_GNN_ARCH,
    ) -> None:
        super().__init__()
        if obs_dim != GRAPH_OBS_SHAPE[0]:
            raise ValueError(
                f"obs_dim {obs_dim} does not match GRAPH_OBS_SHAPE[0]={GRAPH_OBS_SHAPE[0]}; "
                "the GNN model only accepts the graph observation encoding."
            )
        if action_dim != ACTION_SPACE_SIZE:
            raise ValueError(
                f"action_dim {action_dim} != ACTION_SPACE_SIZE {ACTION_SPACE_SIZE}"
            )
        if arch.node_hidden % arch.n_heads != 0:
            raise ValueError(
                f"node_hidden ({arch.node_hidden}) must be divisible by "
                f"n_heads ({arch.n_heads})"
            )
        if arch.player_hidden % arch.n_heads != 0:
            raise ValueError(
                f"player_hidden ({arch.player_hidden}) must be divisible by "
                f"n_heads ({arch.n_heads})"
            )
        if arch.value_kind not in ("scalar", "vector"):
            raise ValueError(
                f"unknown value_kind: {arch.value_kind!r}; "
                "expected 'scalar' or 'vector'"
            )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.arch = arch

        # Static graph adjacency. Registered as buffers so .to(device) moves them.
        edge_index = torch.from_numpy(GRAPH_STRUCTURE.edge_index).long()
        edge_type = torch.from_numpy(GRAPH_STRUCTURE.edge_type).long()
        self.register_buffer("edge_index_static", edge_index, persistent=False)
        edge_type_oh = torch.zeros(
            edge_type.numel(), N_EDGE_TYPES, dtype=torch.float32
        )
        edge_type_oh.scatter_(1, edge_type.view(-1, 1), 1.0)
        self.register_buffer("edge_type_one_hot", edge_type_oh, persistent=False)

        # Scatter indices for assembling the (B, ACTION_SPACE_SIZE) logits.
        for head, name in (
            (ActionHead.ROAD_EDGE, "road_indices"),
            (ActionHead.SETTLE_VERTEX, "settle_indices"),
            (ActionHead.CITY_VERTEX, "city_indices"),
            (ActionHead.MOVE_ROBBER_TILE, "robber_indices"),
            (ActionHead.STEAL_SEAT, "steal_indices"),
            (ActionHead.GLOBAL, "global_indices"),
        ):
            self.register_buffer(
                name,
                torch.from_numpy(ACTION_ROUTING.indices_for(head).copy()).long(),
                persistent=False,
            )

        # ------------------------------------------------------------------
        # Player stream
        # ------------------------------------------------------------------

        self.player_in = orthogonal_init(
            nn.Linear(PLAYER_FEAT_DIM, arch.player_hidden), gain=TRUNK_GAIN
        )
        self.player_attn = nn.TransformerEncoderLayer(
            d_model=arch.player_hidden,
            nhead=arch.n_heads,
            dim_feedforward=arch.player_hidden * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # ------------------------------------------------------------------
        # Per-type input projections
        # ------------------------------------------------------------------

        self.tile_proj = orthogonal_init(
            nn.Linear(TILE_FEAT_DIM, arch.node_hidden), gain=TRUNK_GAIN
        )
        self.vertex_proj = orthogonal_init(
            nn.Linear(
                VERTEX_FEAT_DIM + arch.player_hidden, arch.node_hidden
            ),
            gain=TRUNK_GAIN,
        )
        self.edge_proj = orthogonal_init(
            nn.Linear(
                EDGE_FEAT_DIM + arch.player_hidden, arch.node_hidden
            ),
            gain=TRUNK_GAIN,
        )

        # ------------------------------------------------------------------
        # GNN trunk
        # ------------------------------------------------------------------

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        head_dim = arch.node_hidden // arch.n_heads
        for _ in range(arch.n_mp_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=arch.node_hidden,
                    out_channels=head_dim,
                    heads=arch.n_heads,
                    edge_dim=N_EDGE_TYPES,
                    concat=True,
                    add_self_loops=True,
                )
            )
            self.gnn_norms.append(nn.LayerNorm(arch.node_hidden))

        # ------------------------------------------------------------------
        # Heads
        # ------------------------------------------------------------------

        self.road_head = orthogonal_init(
            nn.Linear(arch.node_hidden, 1), gain=POLICY_HEAD_GAIN
        )
        self.settle_head = orthogonal_init(
            nn.Linear(arch.node_hidden, 1), gain=POLICY_HEAD_GAIN
        )
        self.city_head = orthogonal_init(
            nn.Linear(arch.node_hidden, 1), gain=POLICY_HEAD_GAIN
        )
        self.robber_head = orthogonal_init(
            nn.Linear(arch.node_hidden, 1), gain=POLICY_HEAD_GAIN
        )
        self.steal_head = orthogonal_init(
            nn.Linear(arch.player_hidden, 1), gain=POLICY_HEAD_GAIN
        )

        pooled_dim = (
            arch.node_hidden + arch.player_hidden * N_PLAYERS + GLOBAL_FEAT_DIM
        )
        self.global_trunk = nn.Sequential(
            orthogonal_init(
                nn.Linear(pooled_dim, arch.global_mlp_hidden), gain=TRUNK_GAIN
            ),
            nn.Tanh(),
            orthogonal_init(
                nn.Linear(arch.global_mlp_hidden, arch.global_mlp_hidden),
                gain=TRUNK_GAIN,
            ),
            nn.Tanh(),
        )
        self.global_head = orthogonal_init(
            nn.Linear(arch.global_mlp_hidden, N_GLOBAL_ACTIONS),
            gain=POLICY_HEAD_GAIN,
        )
        value_out_dim = N_PLAYERS if arch.value_kind == "vector" else 1
        self.value_head = orthogonal_init(
            nn.Linear(arch.global_mlp_hidden, value_out_dim), gain=VALUE_HEAD_GAIN
        )

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def value_kind(self) -> ValueKind:
        """Shape of this model's value head; see :data:`ValueKind`."""
        return self.arch.value_kind

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        """Compute masked logits and value for a batch of graph observations.

        ``obs`` is ``(B, GRAPH_OBS_SHAPE[0])`` float; ``mask`` is
        ``(B, ACTION_SPACE_SIZE)`` boolean. Returned logits already have
        illegal entries set to :data:`MASK_FILL_VALUE`. The returned value
        tensor has shape ``(B,)`` for ``value_kind="scalar"`` or
        ``(B, N_PLAYERS)`` for ``value_kind="vector"`` (slot 0 = the
        encoder's viewer; remaining slots are opponents in turn-order
        offset from the viewer).
        """
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"obs shape {tuple(obs.shape)} incompatible with obs_dim={self.obs_dim}"
            )
        if mask.shape != (obs.size(0), self.action_dim):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with "
                f"(batch={obs.size(0)}, action_dim={self.action_dim})"
            )

        B = obs.size(0)
        tiles_x, vertices_x, edges_x, players_x, global_x = self._unpack(obs)

        # ---- Player stream
        player_h0 = self.player_in(players_x)  # (B, 4, h_p)
        player_emb = self.player_attn(player_h0)  # (B, 4, h_p)

        # ---- Owner-aware vertex/edge inputs
        vert_owner = self._gather_vertex_owner(vertices_x, player_emb)
        edge_owner = self._gather_edge_owner(edges_x, player_emb)

        tile_h = self.tile_proj(tiles_x)  # (B, 19, h)
        vert_h = self.vertex_proj(
            torch.cat([vertices_x, vert_owner], dim=-1)
        )  # (B, 54, h)
        edge_h = self.edge_proj(
            torch.cat([edges_x, edge_owner], dim=-1)
        )  # (B, 72, h)

        # ---- GNN trunk (batched message passing)
        node_h = torch.cat([tile_h, vert_h, edge_h], dim=1)  # (B, 145, h)
        node_h = self._run_gnn_trunk(node_h)

        # Slice back per type
        tile_emb = node_h[:, :N_TILES, :]  # (B, 19, h)
        vert_emb = node_h[:, N_TILES : N_TILES + N_VERTICES, :]  # (B, 54, h)
        edge_emb = node_h[:, N_TILES + N_VERTICES :, :]  # (B, 72, h)

        # ---- Pooled context for global head + value
        board_pooled = node_h.mean(dim=1)  # (B, h)
        player_flat = player_emb.reshape(B, -1)  # (B, 4*h_p)
        global_in = torch.cat(
            [board_pooled, player_flat, global_x], dim=-1
        )
        global_trunk_out = self.global_trunk(global_in)

        # ---- Per-head logits
        road_logits = self.road_head(edge_emb).squeeze(-1)  # (B, 72)
        settle_logits = self.settle_head(vert_emb).squeeze(-1)  # (B, 54)
        city_logits = self.city_head(vert_emb).squeeze(-1)  # (B, 54)
        robber_logits = self.robber_head(tile_emb).squeeze(-1)  # (B, 19)
        steal_logits = self.steal_head(player_emb).squeeze(-1)  # (B, 4)
        global_logits = self.global_head(global_trunk_out)  # (B, N_GLOBAL)

        # ---- Scatter into canonical action-space layout
        logits = torch.empty(
            B, ACTION_SPACE_SIZE, device=obs.device, dtype=obs.dtype
        )
        logits.index_copy_(1, self.road_indices, road_logits)
        logits.index_copy_(1, self.settle_indices, settle_logits)
        logits.index_copy_(1, self.city_indices, city_logits)
        logits.index_copy_(1, self.robber_indices, robber_logits)
        logits.index_copy_(1, self.steal_indices, steal_logits)
        logits.index_copy_(1, self.global_indices, global_logits)

        logits = logits.masked_fill(~mask.bool(), MASK_FILL_VALUE)
        value_raw = self.value_head(global_trunk_out)
        # Scalar mode collapses the (B, 1) head down to (B,); vector mode
        # exposes the full (B, N_PLAYERS) per-seat output verbatim.
        value = value_raw.squeeze(-1) if self.arch.value_kind == "scalar" else value_raw
        return ModelOutput(logits=logits, value=value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack(
        self, obs: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Slice the flat obs into the five per-type tensors."""
        B = obs.size(0)
        tiles_x = obs[:, TILE_OFFSET : TILE_OFFSET + TILE_BLOCK].view(
            B, N_TILES, TILE_FEAT_DIM
        )
        vertices_x = obs[:, VERTEX_OFFSET : VERTEX_OFFSET + VERTEX_BLOCK].view(
            B, N_VERTICES, VERTEX_FEAT_DIM
        )
        edges_x = obs[:, EDGE_OFFSET : EDGE_OFFSET + EDGE_BLOCK].view(
            B, N_EDGES, EDGE_FEAT_DIM
        )
        players_x = obs[:, PLAYER_OFFSET : PLAYER_OFFSET + PLAYER_BLOCK].view(
            B, N_PLAYERS, PLAYER_FEAT_DIM
        )
        global_x = obs[:, GLOBAL_OFFSET : GLOBAL_OFFSET + GLOBAL_BLOCK]
        return tiles_x, vertices_x, edges_x, players_x, global_x

    def _gather_vertex_owner(
        self, vertices_x: torch.Tensor, player_emb: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-vertex owner embedding (zero for unowned)."""
        settle = vertices_x[
            :,
            :,
            VERTEX_SETTLE_SLOT_OFFSET : VERTEX_SETTLE_SLOT_OFFSET + N_PLAYERS,
        ]  # (B, 54, 4)
        city = vertices_x[
            :,
            :,
            VERTEX_CITY_SLOT_OFFSET : VERTEX_CITY_SLOT_OFFSET + N_PLAYERS,
        ]  # (B, 54, 4)
        seat_one_hot = settle + city  # (B, 54, 4); 0 for unowned
        return torch.einsum("bvs,bsh->bvh", seat_one_hot, player_emb)

    def _gather_edge_owner(
        self, edges_x: torch.Tensor, player_emb: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-edge owner embedding (zero for unowned)."""
        road = edges_x[
            :,
            :,
            EDGE_ROAD_SLOT_OFFSET : EDGE_ROAD_SLOT_OFFSET + N_PLAYERS,
        ]  # (B, 72, 4)
        return torch.einsum("bes,bsh->beh", road, player_emb)

    def _run_gnn_trunk(self, node_h: torch.Tensor) -> torch.Tensor:
        """Apply ``arch.n_mp_layers`` GATv2 layers under PyG's block-diagonal batching.

        The static ``edge_index`` is tiled across the batch dim with the
        per-sample node offset, giving a single big disconnected graph that
        PyG processes in one pass. Output is reshaped back to ``(B, N, h)``.
        """
        B, N, _ = node_h.shape
        node_h_flat = node_h.reshape(B * N, -1)
        edge_index_batched, edge_attr_batched = self._batched_edges(B, N)

        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            updated = layer(
                node_h_flat,
                edge_index_batched,
                edge_attr=edge_attr_batched,
            )
            node_h_flat = norm(node_h_flat + updated)
        return node_h_flat.view(B, N, -1)

    def _batched_edges(
        self, B: int, N: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile the static edge_index / edge_attr ``B`` times with offsets.

        Built fresh each forward pass; for typical (B, E) this is a couple
        of microseconds and avoids ambiguous max-batch-size caching.
        """
        ei = self.edge_index_static  # (2, E)
        et = self.edge_type_one_hot  # (E, n_edge_types)
        E = ei.size(1)
        # offsets shape (B, 1, 1) broadcasting to (B, 2, E)
        offsets = (
            torch.arange(B, device=ei.device, dtype=ei.dtype).view(B, 1, 1) * N
        )
        # (B, 2, E) → (2, B*E)
        ei_b = (ei.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, B * E)
        et_b = et.unsqueeze(0).expand(B, -1, -1).reshape(B * E, -1)
        return ei_b, et_b
