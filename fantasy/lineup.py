"""Daily lineup filling: start the most valuable players that fit the active slots."""
from .config import BENCH_SLOTS


def active_slots(slot_counts: dict[int, int]) -> list[int]:
    """Expand {slot_id: count} into a list of active (scoring) slot ids."""
    return [s for s, n in sorted(slot_counts.items()) if s not in BENCH_SLOTS for _ in range(n)]


def fill_lineup(candidates: list[tuple[int, float, list[int]]], slots: list[int]) -> set[int]:
    """Pick starters for one day.

    candidates: (player_id, value, eligible_slot_ids) for players with a game.
    Greedy by value with augmenting paths: each player is added if some
    re-shuffle of already-placed players frees a slot. For this kind of
    assignment problem (a transversal matroid) that yields the max-value set.
    """
    holder: dict[int, int] = {}  # slot index -> player_id
    eligible = {pid: set(el) for pid, _, el in candidates}

    def place(pid, seen):
        for i, s in enumerate(slots):
            if s in eligible[pid] and i not in seen:
                seen.add(i)
                if i not in holder or place(holder[i], seen):
                    holder[i] = pid
                    return True
        return False

    for pid, _, _ in sorted(candidates, key=lambda c: -c[1]):
        if len(holder) == len(slots):
            break
        place(pid, set())
    return set(holder.values())
