from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from gca.project_types import Real, Bound
from gca.project_types import Vector

StateType = TypeVar("StateType", bound="State")

class State(Generic[StateType], ABC):  # pyright: ignore [reportInvalidTypeForm]

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        pass

    @abstractmethod
    def next_state(self, neighbors_states: list[StateType]) -> StateType:
        pass


class Cell(Generic[StateType], ABC):

    def __init__(self,
        id_: int,
        index: Vector,
        state: State
    ) -> None:
        self.id = id_
        self.index = index
        self.dimension = len(index)
        self.state = state

    @abstractmethod
    def update(self, n_map: "NeighborMap") -> None:
        pass

    def __getitem__(self, key: int) -> Real:
        return self.index[key]

    def __eq__(self, other):
        return self.id == other.id


CellType = TypeVar('CellType', bound=Cell)
class NeighborMap(Generic[CellType], ABC):

    def __init__(self,dimension: int) -> None:
        self.dimension = dimension

    @abstractmethod
    def map(self, cell: CellType) -> None:
        """
        Maps a new Cell or updates its location in case it is already indexed.
        :param cell: the WolfCell you want to map
        :raise ValueError: if the cell has a different dimension than the map
        """

    @abstractmethod
    def __getitem__(self, args: tuple[CellType, int]) -> list[CellType]:
        """
        Accepts arguments like instance[t, n], where t is the cell whose
        neighboorhood you are looking for and is the number of neighbors
        :raise KeyError if cell is not in this map
        """

    @abstractmethod
    def __contains__(self, cell: CellType) -> bool:
        pass

class CellularAutomaton(Generic[CellType, StateType], ABC):

    def __init__(self,
        index_bound: Bound,
        dimension: int,
        states: list[StateType],
        n_map: NeighborMap[CellType]
    ) -> None:
        CellularAutomaton._verify_dimension(dimension, index_bound)
        self._index_bound = index_bound
        self.dimension = dimension
        self.states = states
        self._n_map = n_map

    @staticmethod
    def _verify_dimension(dimension: int, bounds: Bound) -> None:
        if (dimension != len(bounds.lower)
            or dimension != len(bounds.upper)):
            raise ValueError(f"Lower and upper bound must be of dimension {dimension}")

    @property
    @abstractmethod
    def cells(self) -> list[CellType]:
        pass

    @abstractmethod
    def run(self, i: int = 1) -> None:
        """
        Executes a transition in each cell of the automaton
        :param i: number of transitions
        """
        pass
