from abc import abstractmethod, ABC
from inspect import signature
from typing import TypeAlias, Callable, Sequence, TypeVar, Optional

import numpy as np

from gca.algorithms.gwo import GWOContext
from gca.project_types import Real, Vector, ObjFunc, Solution, MetaHeuristic, SearchAgentRecord
from gca.utils import get_qnt_args, verify_bounds, adjust_to_bounds

T = TypeVar("T")
Neighborhood: TypeAlias = list["WolfCell"]

class WolfCell:

	# ACCESS
	def __init__(self, id_: int, index: Vector, obj_func: ObjFunc, is_unpacking_needed: bool):
		self.id = id_
		self._is_unpacking_needed = is_unpacking_needed
		self._obj_func = obj_func
		self._index: Vector = index
		self._verify_index_dimension(index)

		if self._is_unpacking_needed:
			self.state = obj_func(*index)
		else:
			self.state = obj_func(index.as_np_array())

	@property
	def index(self) -> Vector: return self._index

	@index.setter
	def index(self, value: Vector) -> None:
		self._verify_index_dimension(value)
		self._index = value # type: ignore
		if self._is_unpacking_needed:
			self.state = self._obj_func(*value)
		else:
			self.state = self._obj_func(value.as_np_array())

	## METHODS
	def _verify_index_dimension(self, index: Vector):
		if not self._is_unpacking_needed: return

		if len(signature(self._obj_func).parameters) != self.index.dimension:
			msg = (f"Function {self._obj_func} has a different number of arguments than " + 
				   f"the cell's index dimension {index}")
			raise TypeError(msg)
		
	def __getitem__(self, key: int) -> Real:
		return self.index[key]

	def update_state(self, n_map: "NeighborMap", transition_func: Callable[[Vector, Sequence[Vector]], Vector]):
		neighbors: list[Vector] = [x._index for x in n_map[self, 8]]
		self.index = transition_func(self.index, neighbors)
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


class CAGWO(MetaHeuristic):

	# ACCESS
	def __init__(self,
	 	qnt_cells: int,
		obj_func: ObjFunc,
		dimension: Optional[int],
		n_map: "NeighborMap",
		lower_bound: tuple[Real, ...] | tuple[float, ...],
		upper_bound: tuple[Real, ...] | tuple[float, ...],
		max_interations: int
	):
		self.qnt_cells = qnt_cells
		self.grid_dimension: int = get_qnt_args(obj_func) if dimension is None else dimension
		self.upper_bound, self.lower_bound = verify_bounds(lower_bound, upper_bound, self.grid_dimension)


		# Check if unpacking is needed for obj. function based on whether a dimension is provided
		is_unpacking_needed: bool = dimension is None
		self._cells: list[WolfCell] = self._generate_cells(qnt_cells, self.grid_dimension, obj_func,
														   is_unpacking_needed, lower_bound, upper_bound)

		self._alpha, self._beta, self._delta = self._init_three_best_solutions(self._cells)
		self._n_map: "NeighborMap" = self._map_cells(n_map, self._cells)
		self._gwo_context = GWOContext(max_interations)

	@staticmethod
	def _generate_cells(
		qnt_cells: int,
		grid_dimension: int,
		obj_func: ObjFunc,
		is_unpacking_needed: bool,
		lower_bound: tuple[Real, ...] | tuple[float, ...],
		upper_bound: tuple[Real, ...] | tuple[float, ...],
	) -> list[WolfCell]:
		if qnt_cells < 3: qnt_cells = 3

		random_indexes: list[Vector] = []
		for i in range(qnt_cells):
			random_indexes.append(Vector(np.random.uniform(lower_bound, upper_bound, grid_dimension)))

		return [WolfCell(i, index, obj_func, is_unpacking_needed) for i, index in enumerate(random_indexes)]

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
	def current_search_agents(self) -> tuple[SearchAgentRecord, ...]:
		return tuple([SearchAgentRecord(cell.id, cell.state, cell.index) for cell in self._cells])

	# METHODS
	def _distance(self, this: Vector, other: Vector) -> Vector:
		return abs((other * self._gwo_context.coeff_C) - this)

	def _next_position(self, this: Vector, other: Vector) -> Vector:
		return other - (self._distance(this, other) * self._gwo_context.coeff_A)

	def _real_next_position(self, vector: Vector, neighbors: Sequence[Vector]):
		to_alpha: Vector = self._next_position(vector, self._alpha.index)
		to_beta: Vector = self._next_position(vector, self._beta.index)
		to_delta: Vector = self._next_position(vector, self._delta.index)
		to_neighbors: list[Vector] = [self._next_position(vector, neighbor) for neighbor in neighbors]

		alpha_weight = Real(0.30)
		beta_weight = Real(0.25)
		delta_weight = Real(0.20)
		neighbors_weight = Real(0.25 / len(neighbors))

		to_neighbors_sum: Vector = Vector.sum_vectors([position * neighbors_weight for position in to_neighbors])

		next_pos: Vector = (Vector.sum_vectors((to_alpha * alpha_weight, to_beta * beta_weight,
												to_delta * delta_weight, to_neighbors_sum)))

		return adjust_to_bounds(self.lower_bound, self.upper_bound, next_pos)

	def run(self, i: int = 1) -> None:
		for _ in range(0, i):
			if not self._gwo_context.in_progress(): return

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
			Solution(self._alpha.state, tuple(self._alpha.index)),
			Solution(self._beta.state, tuple(self._beta.index)),
			Solution(self._delta.state, tuple(self._delta.index))
		)
		
