from collections import defaultdict

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class AnyCondition:
    def __init__(self, conditions) -> None:
        self.conditions = conditions

    def reset(self, info: dict):
        for cond in self.conditions:
            cond.reset(info)

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        combined_dones = {agent: False for agent in agents}
        for condition in self.conditions:
            dones = condition.is_done(agents, state)
            for agent, done in dones.items():
                combined_dones[agent] |= done

        return combined_dones


class TimeoutCondition:
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

    def reset(self, info: dict): ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        time_elapsed = state.tick_count / TICKS_PER_SECOND
        done = time_elapsed >= self.timeout_seconds
        return {agent: done for agent in agents}


class NoTouchTimeoutCondition:
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.last_touch_tick = 0

    def reset(self, info: dict):
        self.last_touch_tick = 0

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        if any(car.ball_touches > 0 for car in state.cars.values()):
            self.last_touch_tick = state.tick_count
            done = False
        else:
            time_elapsed = (state.tick_count - self.last_touch_tick) / TICKS_PER_SECOND
            done = time_elapsed >= self.timeout_seconds

        return {agent: done for agent in agents}


class BallTouchTermination:
    """Terminate on the ball being touched"""

    def reset(self, info: dict): ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        return {agent: state.cars[agent].ball_touches > 0 for agent in agents}


class BallMinHeight:
    """Terminate on the ball dropping below given height"""

    def __init__(self, height: float, delay: float = 1.0):
        self.height = height
        self.delay = delay

    def reset(self, info: dict): ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        if state.tick_count / TICKS_PER_SECOND < self.delay:
            return {agent: False for agent in agents}
        return {agent: state.ball.position[2] < self.height for agent in agents}


class MaxTouches:
    """Terminate on the ball being touched too much"""

    def __init__(self, touches: int):
        self.touches = touches
        self.car_touches = defaultdict(int)

    def reset(self, info: dict):
        self.car_touches = defaultdict(int)

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        for agent in agents:
            if state.cars[agent].ball_touches > 0:
                self.car_touches[agent] += 1
        return {agent: self.car_touches[agent] > self.touches for agent in agents}


class FullBoost:
    def reset(self, info: dict): ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        return {agent: state.cars[agent].boost_amount >= 100 for agent in agents}


class NegativeY:
    def reset(self, info: dict): ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        done = state.ball.linear_velocity[1] < 0
        return {agent: done for agent in agents}


class GoalCondition:
    """
    A DoneCondition that is satisfied when a goal is scored.
    """

    def reset(self, info: dict) -> None: ...

    def is_done(self, agents: list[str], state: GameState) -> dict[str, bool]:
        return {agent: state.goal_scored for agent in agents}
