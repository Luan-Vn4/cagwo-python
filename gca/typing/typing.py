from typing import TypeAlias, Iterator, overload, Union, Sequence
import numpy as np
from numpy.typing import NDArray


Real: TypeAlias = np.float64


class Vector:

    def __init__(self, elements: NDArray[Real] | list[float]) -> None:
        super().__init__()
        self._vector: NDArray[Real]
        if isinstance(elements, list):
            self._vector = np.array(elements)
        else:
            self._vector = elements


    @property
    def dimension(self) -> int:
        return self._vector.shape[0]

    def __getitem__(self, key: int) -> Real:
        return Real(self._vector[key])

    def __iter__(self) -> Iterator[Real]:
        return iter(self._vector)

    @overload
    def __add__(self, other: "Vector") -> "Vector":
        ...

    @overload
    def __add__(self, other: Real) -> "Vector":
        ...

    def __add__(self, other: Union["Vector", Real]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(self._vector + other._vector)
        else:
            return Vector(self._vector + other)

    @overload
    def __sub__(self, other: "Vector") -> "Vector":
        ...

    @overload
    def __sub__(self, other: Real) -> "Vector":
        ...

    def __sub__(self, other: Union["Vector", Real]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(self._vector - other._vector)
        else:
            return Vector(self._vector - other)

    @overload
    def __mul__(self, other: "Vector") -> "Vector":
        ...

    @overload
    def __mul__(self, other: Real) -> "Vector":
        ...

    def __mul__(self, other: Union["Vector", Real]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(self._vector * other._vector)
        else:
            return Vector(self._vector * other)

    @overload
    def __truediv__(self, other: "Vector") -> "Vector":
        ...

    @overload
    def __truediv__(self, other: Real) -> "Vector":
        ...

    def __truediv__(self, other: Union["Vector", Real]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(self._vector / other._vector)
        else:
            return Vector(self._vector / other)

    def __matmul__(self, other: "Vector") -> Real:
        return np.dot(self._vector, other._vector)

    def hammard_distance(self, other: "Vector") -> int:
        return np.sum(self._vector != other._vector)

    def __abs__(self) -> "Vector":
        return Vector(abs(self._vector))

    def __str__(self) -> str:
        return str(self._vector)

    def __repr__(self):
        return str(self)

    def __eq__(self, other: "Vector") -> bool:
        if other.dimension != self.dimension: return False

        for i in range(len(self._vector)):
            if self[i] != other[i]: return False

        return True

    def __hash__(self) -> int:
        return hash(tuple(self._vector))

    def __len__(self) -> int:
        return self._vector.shape[0]

    @staticmethod
    def zero_vector(dimension: int) -> "Vector":
        return Vector(np.array([Real(0)] * dimension))

    @staticmethod
    def sum_vectors(vectors: Sequence["Vector"]):
        result: Vector = Vector.zero_vector(vectors[0].dimension)
        for vector in vectors:
            result += vector
        return result
