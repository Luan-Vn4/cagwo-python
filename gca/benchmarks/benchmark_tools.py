from typing import Callable, TypeAlias
from pandas import DataFrame
from gca.benchmarks.benchmark_functions import FunctionRecord
from gca.project_types import Real, MetaHeuristic, ObjFunc
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


Bound: TypeAlias = tuple[float, ...]
LowerBound = Bound
UpperBound = Bound
Dimension: TypeAlias = int
Iterations: TypeAlias = int
MetaheuristicProvider: TypeAlias = Callable[[ObjFunc, Dimension, LowerBound, UpperBound,
                                             Iterations], MetaHeuristic]


@dataclass(frozen=True)
class ObjFunctionBenchmark:
    function: FunctionRecord
    solutions_avg: Real
    solutions_std: Real
    solutions_min: Real
    solutions_max: Real
    solutions_median: Real
    solutions_avg_per_iteration: tuple[Real, ...]


@dataclass(frozen=True)
class MetaHeuristicBenchmark:
    name: str
    executions: int
    iterations: int
    functions_benchmarks: tuple[ObjFunctionBenchmark, ...]


def run_metaheuristic_benchmark(
    metaheuristic_name: str,
    metaheuristic_provider: MetaheuristicProvider,
    obj_functions: tuple[FunctionRecord, ...],
    executions: int,
    iterations: int,
) -> MetaHeuristicBenchmark:
    functions_benchmarks: list[ObjFunctionBenchmark] = []

    for obj_function in obj_functions:
        dimension: int = obj_function.dimension
        lower_bound: tuple[float, ...] = tuple([obj_function.range[0]] * dimension)
        upper_bound: tuple[float, ...] = tuple([obj_function.range[1]] * dimension)

        solutions: NDArray[Real] = np.empty(executions, dtype=Real)
        best_per_iteration: list[NDArray[Real]] = [np.empty(executions, dtype=Real) for _ in range(iterations)]
        for e in range(executions):
            meta_heuristic: MetaHeuristic = metaheuristic_provider(obj_function.function, obj_function.dimension,
                                                                   lower_bound, upper_bound, iterations)
            for i in range(iterations):
                meta_heuristic.run()
                # Best solution for the i iteration in the e execution
                best_per_iteration[i][e] = meta_heuristic.solutions[0].solution

            # Final solution for the e execution
            solutions[e] = meta_heuristic.solutions[0].solution

        mean_value_per_iteration: list[Real] = [np.mean(best_per_iteration[i]) for i in range(iterations)]
        functions_benchmarks.append(
            ObjFunctionBenchmark(obj_function, np.mean(solutions), np.std(solutions), np.min(solutions),
                                 np.max(solutions), np.median(solutions), tuple(mean_value_per_iteration)))

    return MetaHeuristicBenchmark(metaheuristic_name, executions, iterations, tuple(functions_benchmarks))


def to_dataframe(benchmark: MetaHeuristicBenchmark) -> DataFrame:
    functions_names: list[str] = []
    functions_dimension: list[int] = []
    functions_lower_bound: list[float] = []
    functions_upper_bound: list[float] = []
    functions_avg: list[Real] = []
    functions_std: list[Real] = []
    functions_median: list[Real] = []
    functions_min: list[Real] = []
    functions_max: list[Real] = []

    for func_benchmark in benchmark.functions_benchmarks:
        functions_names.append(func_benchmark.function.name)
        functions_dimension.append(func_benchmark.function.dimension)
        functions_lower_bound.append(func_benchmark.function.range[0])
        functions_upper_bound.append(func_benchmark.function.range[1])
        functions_avg.append(func_benchmark.solutions_avg)
        functions_std.append(func_benchmark.solutions_std)
        functions_median.append(func_benchmark.solutions_median)
        functions_min.append(func_benchmark.solutions_min)
        functions_max.append(func_benchmark.solutions_max)

    return DataFrame({
        "Function": functions_names,
        "Dimension": functions_dimension,
        "LowerBound": functions_lower_bound,
        "UpperBound": functions_upper_bound,
        "AVG": functions_avg,
        "STD": functions_std,
        "MEDIAN": functions_median,
        "MIN": functions_min,
        "MAX": functions_max
    })
