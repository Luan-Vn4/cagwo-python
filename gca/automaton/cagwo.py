from abc import abstractmethod, ABC
from gca.typing.typing import Real, Vector
from typing import TypeAlias, Callable, Sequence, TypeVar, NamedTuple
from inspect import signature
from random import random
import numpy as np
from gca.utils import get_qnt_args, verify_bounds, adjust_to_bounds

T = TypeVar("T")
ObjFunc: TypeAlias = Callable[..., Real]
Neighborhood: TypeAlias = list["WolfCell"]


class WolfCell:

	# ACCESS
	def __init__(self, id_: int, index: Vector, obj_func: ObjFunc):
		self.id = id_
		self._obj_func = obj_func
		self._index: Vector = index
		self._verify_index_dimension(index)
		self.state: Real = obj_func(*self._index)

	@property
	def index(self) -> Vector: return self._index

	@index.setter
	def index(self, value: Vector) -> None:
		self._verify_index_dimension(value)
		self._index = value # type: ignore
		self.state = self._obj_func(*value)

	## METHODS
	def _verify_index_dimension(self, index: Vector):
		if len(signature(self._obj_func).parameters) != self.index.dimension:
			msg = (f"Function {self._obj_func} has a different number of arguments than " + 
				   f"the cell's index dimension {index}")
			raise TypeError(msg)
		
	def __getitem__(self, key: int) -> Real:
		return self.index[key]

	def update_state(self, n_map: "NeighborMap", transition_func: Callable[[Sequence[Vector]], Vector]):
		neighbors: list[Vector] = [x._index for x in n_map[self, 8]]
		self.index = transition_func(neighbors)
		n_map.map(self)

	def __eq__(self, other: "WolfCell") -> bool:
		return self.id == other.id

	def __str__(self):
		return f"WolfCell({self.id}, {self.index})"

	def __repr__(self):
		return str(self)


class NeighborMap(ABC):

	@abstractmethod
	def map(self, cell: "WolfCell") -> None:
		"""
		Maps a new WolfCell or updates its location in case it is already indexed
		:param cell: the WolfCell you want to map
		"""
		pass

	@abstractmethod
	def __getitem__(self, args: tuple["WolfCell", int]) -> list["WolfCell"]:
		"""
		Accepts arguments like <code>instance[t, n]</code>, where <code>t</code> is the vector whose neighboorhood
		you are looking for and <code>n</code> is the number of neighbors
		"""
		pass


class Solution(NamedTuple):
	solution: Real
	args: tuple[Real, ...]


class CAGWO:

	# ACCESS
	def __init__(self,
	 	qnt_cells: int,
		obj_func: ObjFunc,
		n_map: "NeighborMap",
		lower_bound: tuple[Real, ...] | tuple[float, ...],
		upper_bound: tuple[Real, ...] | tuple[float, ...],
		max_interations: int
	):
		self.grid_dimension: int = get_qnt_args(obj_func)
		self.upper_bound, self.lower_bound = verify_bounds(lower_bound, upper_bound, self.grid_dimension)
		self._cells: list[WolfCell] = self._generate_cells(qnt_cells, self.grid_dimension, obj_func,
														   lower_bound, upper_bound)
		self._alpha, self._beta, self._delta = self._init_three_best_solutions(self._cells)
		self._n_map: "NeighborMap" = self._map_cells(n_map, self._cells)
		self._coef_a: float = 1
		self.iteration = 0
		self.max_iterations = max_interations


	@staticmethod
	def _generate_cells(
		qnt_cells: int,
		grid_dimension: int,
		obj_func: ObjFunc,
		lower_bound: tuple[Real, ...] | tuple[float, ...],
		upper_bound: tuple[Real, ...] | tuple[float, ...],
	) -> list[WolfCell]:
		if qnt_cells < 3: qnt_cells = 3

		random_indexes: list[Vector] = []
		for i in range(qnt_cells):
			random_indexes.append(Vector(np.random.uniform(lower_bound, upper_bound, grid_dimension)))

		return [WolfCell(i, index, obj_func) for i, index in enumerate(random_indexes)]

	@staticmethod
	def _map_cells(n_map: "NeighborMap", cells: list["WolfCell"]) -> "NeighborMap":
		for cell in cells: n_map.map(cell)
		return n_map

	@staticmethod
	def _init_three_best_solutions(cells: list[WolfCell]) -> tuple[WolfCell, WolfCell, WolfCell]:
		alpha, beta, delta = cells[:3]

		for cell in cells:
			if cell.state < alpha.state:
				delta = beta
				beta = alpha
				alpha = cell
			elif cell.state < beta.state:
				delta = beta
				beta = cell
			elif cell.state < delta.state:
				delta = cell

		return alpha, beta, delta
	
	@property
	def _coef_A(self) -> Real:
		return Real((2 * self._coef_a * random()) - self._coef_a)

	@property
	def _coef_C(self) -> Real:
		return Real(2 * random())

	# METHODS
	def _update_coef_a(self) -> None:
		self._coef_a = 1 - self.iteration/self.max_iterations

	def _distance(self, this: Vector, other: Vector) -> Vector:
		return abs((other * self._coef_C) - this)

	def _next_position(self, this: Vector, other: Vector) -> Vector:
		return other - (self._distance(this, other) * self._coef_A)

	def _real_next_position(self, neighbors: Sequence[Vector]):
		size: int = len(neighbors) + 3

		alpha: Vector = self._alpha.index
		beta: Vector = self._beta.index
		delta: Vector = self._delta.index

		alpha_weight = Real(size * 0.30)
		beta_weight = Real(size * 0.20)
		delta_weight = Real(size * 0.10)
		neighbors_weight = Real(size * (0.40 / len(neighbors)))

		neighbors_sum: Vector = Vector.sum_vectors([neighbor * neighbors_weight for neighbor in neighbors])

		next_pos: Vector = (Vector.sum_vectors((alpha * alpha_weight, beta * beta_weight, delta * delta_weight,
							neighbors_sum)) / Real(size))

		return adjust_to_bounds(self.lower_bound, self.upper_bound, next_pos)

	def run(self, i: int) -> None:
		for _ in range(0, i):
			if self.iteration+1 >= self.max_iterations: return

			self.iteration += 1
			self._update_coef_a()

			for cell in self._cells:
				cell.update_state(self._n_map, self._real_next_position)

				if cell.state < self._alpha.state:
					self._delta = self._beta
					self._beta = self._alpha
					self._alpha = cell
					continue

				if cell.state < self._beta.state:
					self._delta = self._beta
					self._beta = cell
					continue

				if cell.state < self._delta.state:
					self._delta = cell

	@property
	def solutions(self) -> tuple[Solution, Solution, Solution]:
		"""
		Returns the three best solutions at the moment
		"""

		return (
			Solution(self._alpha.state, tuple(self._alpha.index)),
			Solution(self._beta.state, tuple(self._beta.index)),
			Solution(self._delta.state, tuple(self._delta.index))
		)
		
