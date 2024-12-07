import numpy as np
from gca.automaton.cagwo import WolfCell, NeighborMap
from gca.project_types.math_typing import Vector


class LSHNeighborMap(NeighborMap):

    def __init__(self, dimension: int, nbits: int = 64) -> None:
        super().__init__()
        self.dimension = dimension
        self._normals: list[Vector]  = [Vector(np.random.normal(0, 1, dimension)) for _ in range(nbits)]
        self._hash_bucket_map: dict[Vector, list["WolfCell"]] = dict()
        self._id_bucket_map: dict[int, tuple[Vector, list["WolfCell"]]] = dict()
        self.nbits = nbits

    def __getitem__(self, args: tuple["WolfCell", int]) -> list["WolfCell"]:
        cell, neighborhood_size = args

        if cell.id not in self._id_bucket_map: raise KeyError(f"Cell {cell.id} is not mapped")

        hashed_vector: Vector = dot_prod_hash(cell.index, self._normals)
        bucket = self._hash_bucket_map.get(hashed_vector)

        results: list[WolfCell] = [c for c in bucket if c != cell]

        if len(results) < neighborhood_size:
            hash_bucket_pairs = self._hash_bucket_map.items()
            sorted_hash_bucket_pairs = sorted(hash_bucket_pairs,
                                              key=lambda pair: hashed_vector.hammard_distance(pair[0]),
                                              reverse=True)

            for hash_, bucket in sorted_hash_bucket_pairs:
                if hash_ == hashed_vector: continue

                remaining = neighborhood_size - len(results)
                results.extend(bucket[:remaining])

                if len(results) == neighborhood_size: break

        return results

    def map(self, cell: "WolfCell") -> None:
        if cell.id in self._id_bucket_map:
            previous_hash, bucket = self._id_bucket_map[cell.id]
            bucket.remove(cell)
            if len(bucket) == 0: self._hash_bucket_map.pop(previous_hash)

        hashed_vector: Vector = dot_prod_hash(cell.index, self._normals)
        bucket: list[WolfCell] = self._hash_bucket_map.setdefault(hashed_vector, [])

        bucket.append(cell)
        self._id_bucket_map[cell.id] = (hashed_vector, bucket)

    def __str__(self):
        hash_bucket_pairs = self._hash_bucket_map.items()

        str_result = "LSHNeighborMap { "

        for hash_, bucket in hash_bucket_pairs:
            str_result += f"Bucket({hash_}): {bucket} "

        str_result += "}"

        return str_result

def dot_prod_hash(x: Vector, normals: list[Vector]) -> Vector:
    dot_prods = [x @ normal for normal in normals]
    hashed_result = [int(e > 0) for e in dot_prods]
    return Vector(np.array(hashed_result))