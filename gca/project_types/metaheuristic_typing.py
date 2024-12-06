from typing import TypeAlias, Callable, NamedTuple
from gca.project_types import Real, Vector
from abc import ABC, abstractmethod


ObjFunc: TypeAlias = Callable[..., Real]


class Solution(NamedTuple):
    solution: Real
    args: tuple[Real, ...]


class SearchAgentRecord(NamedTuple):
    id: int
    state: Real
    index: Vector


class MetaHeuristic(ABC):

    @abstractmethod
    def run(self, i: int = 1) -> None:
        pass

    @abstractmethod
    def run_all(self) -> None:
        pass

    @abstractmethod
    def in_progress(self) -> bool:
        pass

    @abstractmethod
    def solutions(self) -> tuple[Solution, Solution, Solution]:
        """
        Returns the three best solutions at the moment
        """
        pass

    @abstractmethod
    def current_search_agents(self) -> tuple[SearchAgentRecord, ...]:
        pass