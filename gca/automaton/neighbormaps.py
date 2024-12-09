import numpy as np
from gca.automaton.cellular_automaton import NeighborMap, CellType
from gca.project_types.math_typing import Vector


class LSHNeighborMap(NeighborMap[CellType]):

    def __init__(self, dimension: int, nbits: int = 64) -> None:
        super().__init__(dimension)
        self._normals: list[Vector]  = [Vector(np.random.normal(0, 1, dimension)) for _ in range(nbits)]
        self._hash_bucket_map: dict[Vector, list["CellType"]] = dict()
        self._id_bucket_map: dict[int, tuple[Vector, list["CellType"]]] = dict()
        self.nbits = nbits

    def __getitem__(self, args: tuple[CellType, int]) -> list[CellType]:
        cell, neighborhood_size = args

        if cell.id not in self._id_bucket_map: raise KeyError(f"Cell {cell.id} is not mapped")

        hashed_vector: Vector = dot_prod_hash(cell.index, self._normals)
        bucket = self._hash_bucket_map.get(hashed_vector)

        results: list[CellType] = [c for c in bucket if c != cell]

        if len(results) < neighborhood_size:
            hash_bucket_pairs = self._hash_bucket_map.items()
            sorted_hash_bucket_pairs = sorted(
                hash_bucket_pairs,
                key=lambda pair: hashed_vector.hammard_distance(pair[0]),
                reverse=True)

            for hash_, bucket in sorted_hash_bucket_pairs:
                if hash_ == hashed_vector: continue

                remaining = neighborhood_size - len(results)
                results.extend(bucket[:remaining])

                if len(results) == neighborhood_size: break

        return results

    def __contains__(self, cell: CellType) -> bool:
        return cell.id in self._id_bucket_map

    def map(self, cell: CellType) -> None:
        if cell.dimension != self.dimension: raise ValueError(
            f"Expected {self.dimension} but got {cell.dimension}")

        if cell.id in self._id_bucket_map:
            previous_hash, bucket = self._id_bucket_map[cell.id]
            bucket.remove(cell)
            if len(bucket) == 0: self._hash_bucket_map.pop(previous_hash)

        hashed_vector: Vector = dot_prod_hash(cell.index, self._normals)
        bucket: list[CellType] = self._hash_bucket_map.setdefault(hashed_vector, [])

        bucket.append(cell)
        self._id_bucket_map[cell.id] = (hashed_vector, bucket)


def dot_prod_hash(x: Vector, normals: list[Vector]) -> Vector:
    dot_prods = [x @ normal for normal in normals]
    hashed_result = [int(e > 0) for e in dot_prods]
    return Vector(np.array(hashed_result))