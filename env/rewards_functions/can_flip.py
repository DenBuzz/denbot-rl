from typing import Any

from rlgym.rocket_league.api import GameState


class CanFlip:
    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[str],
        state: GameState,
        is_terminated: dict[str, bool],
        is_truncated: dict[str, bool],
        shared_info: dict[str, Any],
    ) -> dict[str, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: str, state: GameState) -> float:
        car = state.cars[agent]
        return int(car.has_flip)
