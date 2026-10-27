"""Scales: the size of a run as one name, from three agents to a thousand.

    config = PRESETS["default"].at_scale("medium")          # SwarmConfig.at_scale applies SCALES["medium"]

A scale sets the team size, the working steps, the post size, the board's read page, the free board actions per
round, the per-round read and post budgets, the agents' time limit, the concurrency of a round and whether the agents'
model calls use the provider's batch API. Two values follow from the others: the auto-delivery context bound and
Inspect's message limit. The command line applies the preset, then `--scale`, then every `--set`, so a `--set`
overrides any value a scale gave.

Under a domain's turn budget (SwarmConfig.turn_budget, settings.TURN_BUDGETS) the working steps per agent are the
budget divided by the team size, at least one, in place of the scale's steps_per_agent (SwarmConfig.steps_for).
`steps_by_scale` gives that table; `python -m src budget <lone_run_dir>` prints it from a lone run's turns. At the
large scales the literal rule gives one step per agent: 55 turns over 100, 500 or 1000 agents, and 150 over 500 or
1000.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# Characters of attribution an auto-delivered page needs beyond its posts (SwarmConfig.max_context_chars).
CONTEXT_MARGIN = 2048
# Messages one agent's step adds at most, for the message limit: the model's reply, its tool results and the notices.
MESSAGES_PER_STEP = 6
MESSAGE_MARGIN = 20


@dataclass(frozen=True)
class Scale:
    n_agents: int
    steps_per_agent: int
    max_message_chars: int
    read_page: int
    free_board_actions: int
    time_limit_s: int
    max_concurrency: int
    batch_agents: bool
    reads_per_turn: int
    posts_per_turn: int

    @property
    def max_context_chars(self) -> int:
        """Four pages of the longest posts, plus the attribution margin."""
        return 4 * self.read_page * self.max_message_chars + CONTEXT_MARGIN

    @property
    def message_limit(self) -> int:
        """Inspect's message limit for the sample: every agent's working and free steps, six messages each."""
        return self.n_agents * (self.steps_per_agent + self.free_board_actions) * MESSAGES_PER_STEP + MESSAGE_MARGIN

    def fields(self) -> dict:
        """The SwarmConfig fields this scale sets, derived ones included."""
        return {**asdict(self), "max_context_chars": self.max_context_chars, "message_limit": self.message_limit}


SCALES: dict[str, Scale] = {
    "small": Scale(n_agents=3, steps_per_agent=8, max_message_chars=800, read_page=5, free_board_actions=2,
                   time_limit_s=600, max_concurrency=3, batch_agents=False, reads_per_turn=5, posts_per_turn=2),
    "medium": Scale(n_agents=10, steps_per_agent=60, max_message_chars=4000, read_page=10, free_board_actions=3,
                    time_limit_s=3600, max_concurrency=10, batch_agents=False, reads_per_turn=10, posts_per_turn=3),
    "large": Scale(n_agents=100, steps_per_agent=60, max_message_chars=4000, read_page=10, free_board_actions=3,
                   time_limit_s=7200, max_concurrency=20, batch_agents=True, reads_per_turn=10, posts_per_turn=3),
    "extra_large": Scale(n_agents=500, steps_per_agent=40, max_message_chars=2000, read_page=10, free_board_actions=3,
                         time_limit_s=14400, max_concurrency=50, batch_agents=True, reads_per_turn=10,
                         posts_per_turn=3),
    "gigantic": Scale(n_agents=1000, steps_per_agent=30, max_message_chars=2000, read_page=10, free_board_actions=3,
                      time_limit_s=21600, max_concurrency=50, batch_agents=True, reads_per_turn=10, posts_per_turn=3),
}
SCALE_NAMES = tuple(SCALES)


def budget_steps(budget: int, n_agents: int) -> int:
    """Working steps per agent under a turn budget for the swarm: the budget divided equally, at least one."""
    return max(1, budget // n_agents)


def steps_by_scale(budget: int, names: list[str] | None = None) -> dict[str, dict[str, int]]:
    """For each scale (or the named ones): its team size, its own steps_per_agent and the steps under the budget."""
    return {name: {"n_agents": SCALES[name].n_agents, "scale_steps": SCALES[name].steps_per_agent,
                   "budget_steps": budget_steps(budget, SCALES[name].n_agents)}
            for name in (names or SCALES)}
