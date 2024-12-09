import math
from typing import NamedTuple, Callable
from gca.project_types import Real
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class FunctionRecord:
    name: str
    function: Callable[..., Real]
    dimension: int
    range: tuple[float, float]
    optimal_value: float

def _f1_func(x: NDArray[Real]) -> Real:
    absolute_vector: NDArray[Real] = abs(x)
    return np.sum(absolute_vector) + np.prod(absolute_vector)

F1 = FunctionRecord(
    name='F1',
    function=_f1_func,
    dimension=30,
    range=(-10, 10),
    optimal_value=0
)

def _f2_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    result: Real = Real(0)
    for i in range(dim):
        result += np.floor(x[i] + 0.5)**2

    return result

F2 = FunctionRecord(
    name='F2',
    function=_f2_func,
    dimension=30,
    range=(-100, 100),
    optimal_value=0
)

def _f3_func(x: NDArray[Real]) -> Real:
    dimension = x.shape[0]
    result: Real = Real(0)
    for i in range(1, dimension+1):
        result += i * x[i-1]**4

    return result + np.random.random()

F3 = FunctionRecord(
    name='F3',
    function=_f3_func,
    dimension=30,
    range=(-1.28, 1.28),
    optimal_value=0
)

def _f4_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )

F4 = FunctionRecord(
    name='F4',
    function=_f4_func,
    dimension=30,
    range=(-32, 32),
    optimal_value=0
)

def _f5_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    result: Real = Real(0)
    for i in range(dim):
        result += x[i]**2 - 10 * np.cos(2*np.pi * x[i]) + 10
    return result

F5 = FunctionRecord(
    name='F5',
    function=_f5_func,
    dimension=30,
    range=(-5.12, 5.12),
    optimal_value=0
)

def _f6_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    result: Real = Real(0)
    for i in range(dim):
        result += - x[i] * np.sin(np.sqrt(abs(x[i])))

    return result

F6 = FunctionRecord(
    name='F6',
    function=_f6_func,
    dimension=30,
    range=(-500, 500),
    optimal_value=0
)

def _f7_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    prod: Real = Real(1)
    for i in range(dim):
        prod *= np.cos(x[i] / np.sqrt(i+1))

    return Real(1/4000 * np.sum(x**2) - prod + 1)

F7 = FunctionRecord(
    name='F7',
    function=_f7_func,
    dimension=30,
    range=(-600, 600),
    optimal_value=0
)

def _f8_func(x: NDArray[Real]) -> Real:
    return (
        (np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2)))
        * np.exp(-np.sum(np.sin(np.sqrt(abs(x)))**2))
    )

F8 = FunctionRecord(
    name='F8',
    function=_f8_func,
    dimension=30,
    range=(-10, 10),
    optimal_value=-1
)

def _f9_func(x: NDArray[Real]) -> Real:
    dim: int = x.shape[0]
    result: Real = Real(0)
    for i in range(dim):
        result += np.sin(x[i]) * np.sin((i+1) * x[i]**2 / np.pi)**20

    return Real(-result)

F9 = FunctionRecord(
    name="F9",
    function=_f9_func,
    dimension=30,
    range=(0, np.pi),
    optimal_value=-4.687
)
