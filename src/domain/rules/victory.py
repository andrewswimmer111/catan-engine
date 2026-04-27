"""
Victory points and end-of-action winner checks.
"""

from __future__ import annotations

from domain.enums import DevCardType
from domain.game.state import GameState
from domain.ids import PlayerID


def compute_victory_points(state: GameState, player_id: PlayerID) -> int:
    """
    Full VP: building points + VP dev cards in hand (hidden but counted) +
    special awards.
    """
    p = state.players[player_id]
    s = 0
    s += p.settlements_built
    s += 2 * p.cities_built
    for c, _turn in p.dev_cards_in_hand:
        if c is DevCardType.VICTORY_POINT:
            s += 1
    if state.longest_road_holder == player_id:
        s += 2
    if state.largest_army_holder == player_id:
        s += 2
    return s


def check_winner(state: GameState) -> PlayerID | None:
    """
    Return the winner only if the *current* player has reached 10+ VP.

    A player can only declare victory on their own turn — hidden VP dev cards
    are only revealed then, and the rules forbid winning on someone else's
    turn even if a special-award transfer pushes a non-active player to 10.
    """
    pid = state.current_player
    if compute_victory_points(state, pid) >= 10:
        return pid
    return None


def update_largest_army_award(
    state: GameState,
) -> tuple[PlayerID | None, bool]:
    """
    ``Largest army`` requires at least 3 *played* knights. Same tie-keeping
    rule as :func:`update_longest_road_award` in ``longest_road``.
    """
    pids = state.config.player_ids
    counts: dict[PlayerID, int] = {p: state.players[p].knights_played for p in pids}
    m = max(counts.values()) if counts else 0
    if m < 3:
        new_holder: PlayerID | None = None
    else:
        leaders = {p for p, c in counts.items() if c == m}
        cur = state.largest_army_holder
        if cur is not None and cur in leaders:
            new_holder = cur
        else:
            new_holder = min(leaders)
    changed = new_holder != state.largest_army_holder
    state.largest_army_holder = new_holder
    return new_holder, changed
