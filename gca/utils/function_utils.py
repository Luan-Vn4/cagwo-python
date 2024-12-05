from inspect import signature
from typing import Callable, Any
import numpy as np
from gca.typing import Real, Vector


def get_qnt_args(function: Callable[[Any], Any]) -> int:
    return len(signature(function).parameters)

def verify_bounds(
    lower_bound: tuple[Real, ...],
    upper_bound: tuple[Real, ...],
    problem_dimension: int
) -> tuple[tuple[Real, ...], tuple[Real, ...]]:
    if len(lower_bound) != problem_dimension or len(upper_bound) != problem_dimension:
        raise ValueError(f"Lower and upper bound must be of dimension {problem_dimension}")
    return upper_bound, lower_bound

def adjust_to_bounds(
    lower_bound: tuple[Real, ...] | tuple[float, ...],
    upper_bound: tuple[Real, ...] | tuple[float, ...],
    vector: Vector,
) -> Vector:
    result: list[Real] = [np.float64(0)]*len(vector)

    for i, element in enumerate(vector):
        if element < lower_bound[i]:
            result[i] = lower_bound[i]
        elif element > upper_bound[i]:
            result[i] = upper_bound[i]
        else:
            result[i] = element

    return Vector(result)