from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from ray.rllib.utils.typing import AgentID


class ActionParser(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def reset(self, info: dict) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self) -> dict[AgentID, Space]:
        pass

    @abstractmethod
    def parse_action(self, action_dict: dict[AgentID, np.ndarray]) -> Any:
        pass


class ObsBuilder(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> dict[AgentID, Space]:
        pass

    def reset(self, info: dict) -> None:
        pass

    @abstractmethod
    def build_obs(self, state) -> dict[AgentID, Any]:
        pass


class RewardFunction(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self, info: dict) -> None:
        pass

    @abstractmethod
    def get_reward(self, state) -> dict[AgentID, float]:
        pass


class SimulationSetter(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self, info: dict) -> None:
        pass

    @abstractmethod
    def apply(self, sim) -> Any:
        pass

    @property
    @abstractmethod
    def agents(self) -> list[AgentID]:
        pass


class TerminalCondition(ABC):
    @abstractmethod
    def reset(self, info: dict) -> None:
        pass

    @abstractmethod
    def is_done(self, state) -> dict[AgentID, bool]:
        pass


class Informer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def reset(self, info: dict) -> None:
        pass

    @abstractmethod
    def get_info(self, state) -> dict[AgentID, Any]:
        return {}
