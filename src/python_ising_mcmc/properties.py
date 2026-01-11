import numpy as np
import numpy.typing as npt


def calc_lattice_energy(
    lattice: npt.NDArray[np.int32], coupl_const: float = 1.0
) -> float:
    # Sum nearest-neighbor interactions (using PBCs)
    energy = -coupl_const * np.sum(
        lattice
        * (  # ngl ChatGPT did this and its cracked
            np.roll(lattice, shift=1, axis=0)  # Shift up
            + np.roll(lattice, shift=-1, axis=0)  # Shift down
            + np.roll(lattice, shift=1, axis=1)  # Shift right
            + np.roll(lattice, shift=-1, axis=1)  # Shift left
        )
    )

    return energy / 2  # Each pair counted twice (I think)


def calc_lattice_magnetisation(lattice: npt.NDArray[np.int32]) -> np.int32:
    return np.sum(lattice, dtype=np.int32)
