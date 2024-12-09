from random import uniform
import sys
from abc import ABC, abstractmethod
from random import choice
from typing import Callable, TypeVar, TypeAlias

import numpy as np

from gca.algorithms.gwo import GWOContext
from gca.automaton.cellular_automaton import NeighborMap, CellularAutomaton, Cell, State
from gca.project_types import Vector, ObjFunc, Solution, MetaHeuristic, SearchAgentRecord, Bound, Real
from gca.utils import adjust_to_bounds

# === TYPES ===
StateType = TypeVar('StateType', bound="WolfState")
WolfStateProvider: TypeAlias = Callable[["CAGWOContext"], StateType]


# === STATES ===
class WolfState(State["WolfState"], ABC):

    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        pass

    @abstractmethod
    def default_weight(self) -> float:
        pass

    def __repr__(self):
        return f"WolfState({self.name})"


class ActiveWolfState(WolfState, ABC):

    def __init__(self, name: str, cagwo_context: "CAGWOContext"):
        super().__init__(name)
        self._context = cagwo_context
        self._context.state_context[ActiveWolfState] = self

    def _distance(self, this: Vector, other: Vector) -> Vector:
        return abs((other * self._context.coeff_C) - this)

    def _next_position(self, this: Vector, other: Vector) -> Vector:
        #print(self.name)
        #print(self._context.coeff_a)
        return other - (self._distance(this, other) * self._context.coeff_A)


    def _weighted_expected_state(
        self,
        states: list[WolfState],
        weights: dict[WolfState, float]
    ) -> WolfState:
        expected_weight: dict[WolfState, float] = {
            state: weight*len(states)*uniform(0.5, 1) for state, weight in weights.items()}

        #max_expected_weight = max(expected_weight.values())
        #min_expected_weight = min(expected_weight.values())

        def randomize(weight: float) -> float:
            return (
                weight +
                min(abs(min(weights.values()) - weight), abs(max(weights.values()) - weight)) *
                uniform(-1, 1)
            )

        sum_weight: float = sum([randomize(weights.get(state, 0)) for state in states])

        return min(expected_weight.keys(),
                   key=lambda state: abs(expected_weight[state] - sum_weight))


class InactiveWolfState(WolfState):

    def __init__(self):
        super().__init__("Inactive")

    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        return current_index

    def default_weight(self) -> float:
        return 1

    def next_state(self, neighbors_states: list[WolfState]) -> WolfState:
        return self


class ExploringState(ActiveWolfState):

    def __init__(self, cagwo_context: "CAGWOContext"):
        super().__init__("Exploring", cagwo_context)

    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        to_neighbors: list[Vector] = [self._next_position(current_index, neighbor) for neighbor in neighbors]

        to_neighbors_sum: Vector = Vector.sum_vectors([position for position in to_neighbors])

        next_pos: Vector = to_neighbors_sum / Real(len(neighbors))

        return adjust_to_bounds(self._context.bound.lower,
                                self._context.bound.upper, next_pos)

    def next_state(self, neighbors_states: list[WolfState]) -> WolfState:
        advancing = self._context.get_state(AdvancingState)
        attacking = self._context.get_state(AttackingState)

        state_weights: dict[WolfState, float] = {
            self: 1.0 * self.default_weight(),
            attacking: 1.5 * attacking.default_weight(),
            advancing: 2.0 * advancing.default_weight()
        }

        return self._weighted_expected_state(neighbors_states, state_weights)

    def default_weight(self) -> float:
        return 0.5 * self._context.coeff_a


class AdvancingState(ActiveWolfState):

    def __init__(self, cagwo_context: "CAGWOContext"):
        super().__init__("Advancing", cagwo_context)

    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        to_alpha: Vector = self._next_position(current_index, self._context.alpha.index)
        to_beta: Vector = self._next_position(current_index, self._context.beta.index)
        to_delta: Vector = self._next_position(current_index, self._context.delta.index)
        to_neighbors: list[Vector] = [self._next_position(current_index, neighbor) for neighbor in neighbors]

        alpha_weight = Real(0.40)
        beta_weight = Real(0.30)
        delta_weight = Real(0.20)
        neighbors_weight = Real(0.10 / len(neighbors))

        to_neighbors_sum: Vector = Vector.sum_vectors([position * neighbors_weight for position in to_neighbors])

        next_pos: Vector = (Vector.sum_vectors((to_alpha * alpha_weight, to_beta * beta_weight,
                                                to_delta * delta_weight, to_neighbors_sum)))

        return adjust_to_bounds(self._context.bound.lower,
                                self._context.bound.upper, next_pos)

    def next_state(self, neighbors_states: list[WolfState]) -> WolfState:
        exploring = self._context.get_state(ExploringState)
        attacking = self._context.get_state(AttackingState)

        state_weights: dict[WolfState, float] = {
            self: 1.0 * self.default_weight(),
            exploring: 1.5 * exploring.default_weight(),
            attacking: 2.0 * attacking.default_weight()
        }

        return self._weighted_expected_state(neighbors_states, state_weights)

    def default_weight(self) -> float:
        return 1 / (self._context.coeff_a + sys.float_info.min)


class AttackingState(ActiveWolfState):

    def __init__(self, cagwo_context: "CAGWOContext"):
        super().__init__("Attacking", cagwo_context)

    def next_index(self, current_index: Vector, neighbors: list[Vector]) -> Vector:
        to_alpha: Vector = self._next_position(current_index, self._context.alpha.index)
        to_beta: Vector = self._next_position(current_index, self._context.beta.index)
        to_delta: Vector = self._next_position(current_index, self._context.delta.index)

        next_pos: Vector = Vector.sum_vectors([to_alpha, to_beta, to_delta]) / Real(3)

        return adjust_to_bounds(self._context.bound.lower,
                                self._context.bound.upper, next_pos)

    def next_state(self, neighbors_states: list[WolfState]) -> WolfState:
        exploring = self._context.get_state(ExploringState)
        advancing = self._context.get_state(AdvancingState)

        state_weights: dict[WolfState, float] = {
            self: 1.0 * self.default_weight(),
            advancing: 1.5 * advancing.default_weight(),
            exploring: 2.0 * exploring.default_weight()
        }

        return self._weighted_expected_state(neighbors_states, state_weights)

    def default_weight(self) -> float:
        return 1 / (self._context.coeff_a**2 + sys.float_info.min)


# === COMPONENTS ===
class CAGWOContext(GWOContext):

    def __init__(
        self,
        max_iterations: int,
        bound: Bound,
        alpha: "WolfCell",
        beta: "WolfCell",
        delta: "WolfCell",
    ) -> None:
        super().__init__(max_iterations)
        self.state_context: dict[type[WolfState], WolfState] = dict()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.bound = bound

    def update_leaders(self, wolf: "WolfCell") -> None:
        if self.alpha is None or wolf.value < self.alpha.value:
            self.delta = self.beta
            self.beta = self.alpha
            self.alpha = wolf
            return

        if self.beta is None or wolf.value < self.beta.value:
            self.delta = self.beta
            self.beta = wolf
            return

        if self.delta is None or wolf.value < self.delta.value:
            self.delta = wolf

    def register_state(self, *providers: WolfStateProvider[WolfState]) -> None:
        for provider in providers:
            state: WolfState = provider(self)
            self.state_context[type(state)] = state

    def get_state(self, state_type: type[WolfState]) -> WolfState:
        if state_type not in self.state_context: raise KeyError(
            f"State {state_type} not registered in CAGWOContext")
        return self.state_context[state_type]

    @property
    def states(self) -> list[WolfState]:
        return list(self.state_context.values())


class WolfCell(Cell[WolfState]):

    # ACCESS
    def __init__(self,
        id_: int,
        index: Vector,
        obj_func: ObjFunc,
        state: WolfState = InactiveWolfState(),
    ):
        self.id = id_
        self._obj_func = obj_func
        self._index: Vector = index
        self.value = obj_func(index.as_np_array())
        super().__init__(id_, index, state)

    @property
    def index(self) -> Vector:
        return self._index

    @index.setter
    def index(self, value: Vector) -> None:
        self._index = value  # type: ignore
        self.value = self._obj_func(value.as_np_array())

    def update(self, n_map: NeighborMap["WolfCell"]) -> None:
        neighborhood: list[WolfCell] = n_map[self, 8]
        self.index = self.state.next_index(self.index, [x.index for x in neighborhood])
        self.state = self.state.next_state([neighbor.state for neighbor in neighborhood]
                                           + [self.state])
        n_map.map(self)


# === ALGORITHM/AUTOMATON ===
class CAGWO(CellularAutomaton[WolfCell, WolfState], MetaHeuristic):

    # ACCESS
    def __init__(self,
        qnt_cells: int,
        obj_func: ObjFunc,
        dimension: int,
        map_provider: Callable[[int], "NeighborMap[WolfCell]"],
        bound: Bound,
        max_iterations: int
    ):
        self._cells: list[WolfCell] = self._generate_cells(qnt_cells, dimension,
                                                           obj_func, bound)
        self._context = CAGWOContext(max_iterations, bound, *self._get_three_best())
        self._context.register_state(
            lambda ctx: ExploringState(ctx),
            lambda ctx: AdvancingState(ctx),
            lambda ctx: AttackingState(ctx)
        )

        n_map: "NeighborMap" = map_provider(dimension)
        for cell in self._cells: n_map.map(cell)
        CellularAutomaton.__init__(self, bound, dimension, self._context.states, n_map)
        self._activate_cells_states()

    @staticmethod
    def _generate_cells(
        qnt_cells: int,
        grid_dimension: int,
        obj_func: ObjFunc,
        bound: Bound,
    ) -> list[WolfCell]:
        if qnt_cells < 3: qnt_cells = 3

        random_indexes: list[Vector] = []
        for i in range(qnt_cells):
            random_indexes.append(Vector(np.random.uniform(bound.lower, bound.upper, grid_dimension)))

        return [WolfCell(i, index, obj_func) for i, index in enumerate(random_indexes)]

    def _get_three_best(self) -> tuple[WolfCell, WolfCell, WolfCell]:
        alpha, beta, delta = self._cells[:3]

        for cell in self._cells:
            if cell.value < alpha.value:
                delta = beta
                beta = alpha
                alpha = cell
            elif cell.value < beta.value:
                delta = beta
                beta = cell
            elif cell.value < delta.value:
                delta = cell

        return alpha, beta, delta

    def _activate_cells_states(self) -> None:
        for cell in self._cells:
            cell.state = choice(self.states)

    @property
    def current_search_agents(self) -> tuple[SearchAgentRecord, ...]:
        return tuple([SearchAgentRecord(cell.id, cell.value, cell.index) for cell in self._cells])

    @property
    def cells(self) -> list[WolfCell]:
        return self._cells

    # METHODS
    def run(self, i: int = 1) -> None:
        for _ in range(0, i):
            if not self._context.in_progress(): return

            for cell in self._cells:
                cell.update(self._n_map)

                self._context.update_leaders(cell)

            self._context.increment_iteration()

    def run_all(self):
        self.run(self._context.remaining_iterations())

    def in_progress(self) -> bool:
        return self._context.in_progress()

    @property
    def solutions(self) -> tuple[Solution, Solution, Solution]:
        """
        Returns the three best solutions at the moment
        """
        return (
            Solution(self._context.alpha.value, tuple(self._context.alpha.index)),
            Solution(self._context.beta.value, tuple(self._context.beta.index)),
            Solution(self._context.delta.value, tuple(self._context.delta.index))
        )
