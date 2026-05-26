"""Iteration-cadenced evaluator for :class:`rl.training.alphazero.AlphaZeroTrainer`.

The PPO trainer drives :class:`rl.evaluation.scheduler.EvalScheduler` on
an env-step cadence with three benchmark families (vs random, vs
heuristic, vs sampled-from-pool). AZ wants the same shape but on an
iteration cadence and with a different third benchmark: **vs the prior
snapshot** of the same network — the most informative "am I getting
better?" signal during continuous self-play.

Promotion knob
--------------

When ``promotion_threshold`` is ``None`` (the default, "continuous
training"), the prior-snapshot opponent is unconditionally refreshed to
the latest snapshot every time the trainer calls
:meth:`AZEvaluator.refresh_prior_snapshot`. The eval signal therefore
asks "am I better than I was N iterations ago?".

When ``promotion_threshold`` is set, the evaluator only refreshes the
prior-snapshot after a vs-prior-snapshot eval whose learner win-rate
exceeds the threshold. This is the AlphaGo-Zero-style promotion gate
("only train against versions of yourself you've already proven you
beat"). The current best snapshot's iteration is exposed as
:attr:`prior_best_iter`.

Elo identity
------------

Each eval registers the current learner under ``"learner@iter_{i}"`` and
the opposing baselines under static IDs (``"random"`` / ``"heuristic"``
/ ``"prior_snapshot@iter_{j}"``). The persisted ratings file therefore
shows a chronological Elo trajectory across the run.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.elo import EloTracker
from rl.evaluation.metrics import GameStats, TournamentResult
from rl.evaluation.tournament import Tournament

__all__ = ["AZEvalConfig", "AZEvaluator"]


_DEFAULT_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


@dataclass(frozen=True)
class AZEvalConfig:
    """Knobs for an :class:`AZEvaluator`.

    * ``every_iters`` — only run eval when ``iteration % every_iters == 0``;
      a value ``<= 0`` disables eval entirely (the trainer skips the call).
    * ``n_games`` — games per benchmark per eval.
    * ``bench_*`` — toggle individual benchmark families.
    * ``promotion_threshold`` — optional learner-win-rate gate for refreshing
      the prior-snapshot. ``None`` means continuous training (always refresh
      on the next ``refresh_prior_snapshot`` call); a float in (0, 1] means
      only refresh when the latest vs-prior eval crossed the threshold.
    * ``base_seed`` — base for benchmark RNG; offset by family + iteration so
      different benchmarks see different games but the same iter is
      reproducible.
    """

    every_iters: int = 5
    n_games: int = 20
    bench_random: bool = True
    bench_heuristic: bool = True
    bench_prior_snapshot: bool = True
    promotion_threshold: Optional[float] = None
    base_seed: int = 1_000_000
    player_ids: tuple[PlayerID, ...] = _DEFAULT_PLAYER_IDS


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


class AZEvaluator:
    """Owns the AZ eval cadence, the prior-snapshot opponent, and Elo."""

    def __init__(
        self,
        config: AZEvalConfig,
        env_factory: Callable[[int], CatanEnv],
        elo: EloTracker | None = None,
        ratings_path: Path | None = None,
    ) -> None:
        self._cfg = config
        self._env_factory = env_factory
        self._elo = elo if elo is not None else EloTracker()
        self._ratings_path = Path(ratings_path) if ratings_path else None

        self._prior_snapshot_agent: PolicyAgent | None = None
        self._prior_snapshot_iter: int | None = None
        # Tracks the last computed vs-prior-snapshot win-rate AND the
        # iteration it came from. ``refresh_prior_snapshot`` only honours
        # the promotion gate when the win-rate is from the same iteration
        # the caller is currently snapshotting — a stale win-rate from
        # several iterations ago would gate the wrong learner.
        self._last_vs_prior_win_rate: float | None = None
        self._last_vs_prior_eval_iter: int | None = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def elo(self) -> EloTracker:
        return self._elo

    @property
    def prior_snapshot_iter(self) -> int | None:
        """Iteration at which the current prior-snapshot was captured."""
        return self._prior_snapshot_iter

    @property
    def last_vs_prior_win_rate(self) -> float | None:
        return self._last_vs_prior_win_rate

    # ------------------------------------------------------------------
    # Snapshot refresh
    # ------------------------------------------------------------------

    def refresh_prior_snapshot(self, learner: PolicyAgent, iteration: int) -> bool:
        """Capture ``learner`` as the new prior-snapshot opponent.

        When ``promotion_threshold`` is set AND a prior-snapshot already
        exists, the refresh is gated on the most recent
        vs-prior-snapshot win-rate exceeding the threshold AND coming
        from the same iteration as this refresh — a stale win-rate from
        an earlier iteration would gate the wrong learner. Returns
        whether a refresh actually happened, so the trainer can log the
        event.

        With no prior-snapshot yet (first call) the threshold is bypassed:
        we always seed the prior-snapshot on the first refresh so a later
        eval has something to compare against.
        """
        threshold = self._cfg.promotion_threshold
        if threshold is not None and self._prior_snapshot_agent is not None:
            wr = self._last_vs_prior_win_rate
            wr_iter = self._last_vs_prior_eval_iter
            if wr is None or wr_iter != iteration or wr < threshold:
                return False

        self._prior_snapshot_agent = _freeze_copy(learner)
        self._prior_snapshot_iter = iteration
        return True

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def maybe_run(
        self,
        learner: PolicyAgent,
        iteration: int,
    ) -> dict[str, float] | None:
        """Run the configured benchmarks if ``iteration`` is on cadence.

        Returns ``None`` when no eval ran. Otherwise returns a flat
        ``{name → scalar}`` dict. The trainer is responsible for
        forwarding the dict to TB (single SummaryWriter, consistent
        global-step x-axis).
        """
        if self._cfg.every_iters <= 0:
            return None
        if iteration % self._cfg.every_iters != 0:
            return None

        learner_id = f"learner@iter_{iteration}"
        metrics: dict[str, float] = {}

        if self._cfg.bench_random:
            self._run_anchor(
                learner=learner,
                learner_id=learner_id,
                bench_name="vs_random",
                opponent_factory=_random_opponent_factory(),
                opponent_id="random",
                metrics=metrics,
                iteration=iteration,
                seed_offset=0,
            )
        if self._cfg.bench_heuristic:
            self._run_anchor(
                learner=learner,
                learner_id=learner_id,
                bench_name="vs_heuristic",
                opponent_factory=lambda _rng: HeuristicAgent(),
                opponent_id="heuristic",
                metrics=metrics,
                iteration=iteration,
                seed_offset=1,
            )
        if self._cfg.bench_prior_snapshot and self._prior_snapshot_agent is not None:
            opp_id = f"prior_snapshot@iter_{self._prior_snapshot_iter}"
            self._run_anchor(
                learner=learner,
                learner_id=learner_id,
                bench_name="vs_prior_snapshot",
                opponent_factory=lambda _rng: self._prior_snapshot_agent,  # type: ignore[return-value]
                opponent_id=opp_id,
                metrics=metrics,
                iteration=iteration,
                seed_offset=2,
            )
            self._last_vs_prior_win_rate = metrics["vs_prior_snapshot/win_rate"]
            self._last_vs_prior_eval_iter = iteration

        if self._ratings_path is not None:
            self._elo.save(self._ratings_path)

        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_anchor(
        self,
        *,
        learner: PolicyAgent,
        learner_id: str,
        bench_name: str,
        opponent_factory: Callable[[random.Random], Agent],
        opponent_id: str,
        metrics: dict[str, float],
        iteration: int,
        seed_offset: int,
    ) -> None:
        """Play ``n_games`` of learner vs three ``opponent_factory()`` seats."""
        seed = self._cfg.base_seed + iteration * 1_009 + seed_offset
        rng = random.Random(seed)
        learner_seat = self._cfg.player_ids[0]
        agents: dict[PlayerID, Agent] = {learner_seat: learner}
        for pid in self._cfg.player_ids[1:]:
            agents[pid] = opponent_factory(random.Random(rng.randrange(2**32)))
        result = Tournament(self._env_factory).play(
            agents, n_games=self._cfg.n_games, base_seed=seed
        )

        win_rate = float(result.win_rates.get(learner_seat, 0.0))
        self._update_elo_from_result(result, learner_id, opponent_id, learner_seat)
        metrics[f"{bench_name}/win_rate"] = win_rate
        metrics[f"{bench_name}/elo"] = self._elo.rating(learner_id)
        metrics[f"{bench_name}/mean_turns"] = float(result.mean_turns)

    def _update_elo_from_result(
        self,
        result: TournamentResult,
        learner_id: str,
        opponent_id: str,
        learner_seat: PlayerID,
    ) -> None:
        """Per-game pairwise Elo updates with a 2-agent abstraction.

        Mirrors :meth:`EvalScheduler._update_elo_from_result`: collapse
        the four seats into "learner vs opponent" — the learner scores
        1.0 if it won, 0.0 otherwise; the (single) opponent label
        absorbs the three baseline seats. Same documented wart applies.
        """
        for game in result.games:
            sa = 1.0 if game.winner == learner_seat else 0.0
            sb = 1.0 - sa if game.winner is not None else 0.0
            self._elo.update([learner_id, opponent_id], [sa, sb])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _freeze_copy(learner: PolicyAgent) -> PolicyAgent:
    """Return a deep-copy of ``learner`` whose weights won't track future
    gradient updates. Preserves device and obs/action encoders so the
    copy plays identically to the original at this point in time.

    ``stochastic_play=True`` keeps the prior-snapshot opponent from
    becoming a single-strategy puppet — greedy opponents are easy to
    exploit and produce uninformative win-rate signal.
    """
    new_model = copy.deepcopy(learner.model)
    new_model.eval()
    return PolicyAgent(
        new_model,
        learner.action_encoder,
        obs_encoder=learner.obs_encoder,
        device=learner.device,
        stochastic_play=True,
    )


def _random_opponent_factory() -> Callable[[random.Random], Agent]:
    def factory(rng: random.Random) -> Agent:
        return RandomAgent(rng, skip_proposals=True)

    return factory
