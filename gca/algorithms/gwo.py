from inspect import signature
from random import random
from typing import Optional, Callable
import numpy as np
from gca.project_types import Real, Vector, MetaHeuristic, SearchAgentRecord
from gca.project_types.metaheuristic_typing import ObjFunc, Solution
from gca.utils import get_qnt_args, verify_bounds, adjust_to_bounds


class Wolf:

    def __init__(self, vector: Vector, obj_func: ObjFunc, is_unpacking_needed: bool):
        self._is_unpacking_needed = is_unpacking_needed
        self._obj_func = obj_func
        self._vector: Vector = vector
        self._verify_index_dimension(vector)

        if self._is_unpacking_needed:
            self.value = obj_func(*vector)
        else:
            self.value = obj_func(vector.as_np_array())

    @property
    def vector(self) -> Vector:
        return self._vector

    @vector.setter
    def vector(self, value: Vector) -> None:
        self._verify_index_dimension(value)
        self._vector = value  # type: ignore
        if self._is_unpacking_needed:
            self.value = self._obj_func(*value)
        else:
            self.value = self._obj_func(value.as_np_array())

    def _verify_index_dimension(self, index: Vector):
        if not self._is_unpacking_needed: return

        if len(signature(self._obj_func).parameters) != self.vector.dimension:
            msg = (f"Function {self._obj_func} has a different number of arguments than " +
                   f"the cell's index dimension {index}")
            raise TypeError(msg)

    def move(self, next_position_func: Callable[[Vector], Vector]):
        self.vector = next_position_func(self.vector)


class GWO(MetaHeuristic):

    def __init__(
        self,
        population_size: int,
        obj_func: ObjFunc,
        dimension: Optional[int],
        lower_bound: tuple[Real, ...] | tuple[float, ...],
        upper_bound: tuple[Real, ...] | tuple[float, ...],
        max_iterations
    ) -> None:
        self.population_size = population_size
        self.dimension = get_qnt_args(obj_func) if dimension is None else dimension
        self.upper_bound, self.lower_bound = verify_bounds(lower_bound, upper_bound, self.dimension)

        is_unpacking_needed: bool = dimension is None
        self._wolves: list[Wolf] = GWO._generate_wolves(population_size, dimension, is_unpacking_needed,
                                                        obj_func, lower_bound, upper_bound)

        self._alpha, self._beta, self._delta = self._init_three_best_solutions(self._wolves)
        self._gwo_context = GWOContext(max_iterations)

    @staticmethod
    def _generate_wolves(
            population_size: int,
            dimension: int,
            is_unpacking_needed: bool,
            obj_func: ObjFunc,
            lower_bound: tuple[Real, ...] | tuple[float, ...],
            upper_bound: tuple[Real, ...] | tuple[float, ...],
    ) -> list[Wolf]:
        if population_size < 3: population_size = 3

        random_vectors: list[Vector] = []
        for i in range(population_size):
            random_vectors.append(Vector(np.random.uniform(lower_bound, upper_bound, dimension)))

        return [Wolf(random_vector, obj_func, is_unpacking_needed) for random_vector in random_vectors]

    @staticmethod
    def _init_three_best_solutions(wolves: list[Wolf]) -> tuple[Wolf, Wolf, Wolf]:
        alpha, beta, delta = wolves[:3]

        for wolf in wolves:
            if wolf.value < alpha.value:
                delta = beta
                beta = alpha
                alpha = wolf
            elif wolf.value < beta.value:
                delta = beta
                beta = wolf
            elif wolf.value < delta.value:
                delta = wolf

        return alpha, beta, delta

    @property
    def current_search_agents(self) -> tuple[SearchAgentRecord, ...]:
        return tuple([SearchAgentRecord(i, wolf.value, wolf.vector) for i, wolf in enumerate(self._wolves)])

    def _distance(self, this: Vector, other: Vector) -> Vector:
        return abs((other * self._gwo_context.coeff_C) - this)

    def _next_position(self, this: Vector, other: Vector) -> Vector:
        return other - (self._distance(this, other) * self._gwo_context.coeff_A)

    def _real_next_position(self, vector: Vector):
        to_alpha: Vector = self._next_position(vector, self._alpha.vector)
        to_beta: Vector = self._next_position(vector, self._beta.vector)
        to_delta: Vector = self._next_position(vector, self._delta.vector)

        next_pos: Vector = (Vector.sum_vectors([to_alpha, to_beta, to_delta]) / 3)

        return adjust_to_bounds(self.lower_bound, self.upper_bound, next_pos)

    def run(self, i: int = 1) -> None:
        for _ in range(0, i):
            if not self._gwo_context.in_progress(): return

            for wolf in self._wolves:
                wolf.move(self._real_next_position)

                if wolf.value < self._alpha.value:
                    self._delta = self._beta
                    self._beta = self._alpha
                    self._alpha = wolf
                    continue

                if wolf.value < self._beta.value:
                    self._delta = self._beta
                    self._beta = wolf
                    continue

                if wolf.value < self._delta.value:
                    self._delta = wolf

            self._gwo_context.increment_iteration()

    def run_all(self):
        self.run(self._gwo_context.remaining_iterations())

    def in_progress(self) -> bool:
        return self._gwo_context.in_progress()

    @property
    def solutions(self) -> tuple[Solution, Solution, Solution]:
        """
        Returns the three best solutions at the moment
        """
        return (
            Solution(self._alpha.value, tuple(self._alpha.vector)),
            Solution(self._beta.value, tuple(self._beta.vector)),
            Solution(self._delta.value, tuple(self._delta.vector))
        )


class GWOContext:

    def __init__(self, max_iterations: int, coef_a: int = 2) -> None:
        self._coeff_a = coef_a
        self.iteration = 0
        self.max_iterations = max_iterations

    @property
    def coeff_a(self) -> float:
        return self._coeff_a - self._coeff_a * (self.iteration / self.max_iterations)

    @property
    def coeff_A(self) -> Real:
        return Real((2 * self.coeff_a * random()) - self.coeff_a)

    @property
    def coeff_C(self) -> Real:
        return Real(2 * random())

    def increment_iteration(self) -> None:
        self.iteration += 1

    def remaining_iterations(self) -> int:
        return (self.max_iterations-1) - self.iteration

    def in_progress(self) -> bool:
        return self.iteration < self.max_iterations-1
