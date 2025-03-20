from typing import Any
from rlgym.rocket_league.api import GameState


class BallTouchTermination:
    """Terminate on the ball being touched"""

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def is_done(self, agents: list[str], state: GameState, shared_info: dict[str, Any]) -> dict[str, bool]:
        return {agent: state.cars[agent].ball_touches > 0 for agent in agents}
