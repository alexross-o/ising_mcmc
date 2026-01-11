import gc
import time
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional, Tuple, Any
from .types import MonteCarloResult, IsingMCMCResult
from .properties import calc_lattice_energy


MC_RESULTS = Any


class BaseMonteCarloEngine(ABC):

    @abstractmethod
    def _do_mc_step(self):
        pass

    @abstractmethod
    def _equilibrate(self, n_steps: int):
        pass

    @abstractmethod
    def _simulate(self, n_steps: int) -> MC_RESULTS:
        pass

    @abstractmethod
    def run(self) -> MonteCarloResult:
        pass


class IsingMonteCarloEngine(BaseMonteCarloEngine):

    def __init__(
        self,
        init_lattice: Optional[npt.NDArray[np.int32]] = None,
        lattice_size: Optional[Tuple[int, int]] = (10, 10),
        eq_steps: int = 3000,
        sim_steps: int = 10000,
        kB: float = 1.0,
        coupl_const: float = 1.0,
        temp: float = 2.6,
        flip_frac: float = 0.1,
    ) -> None:

        super().__init__()

        if init_lattice is None:
            self.lattice_size = lattice_size
            self.init_lattice = np.random.choice([-1, 1], size=lattice_size)
        elif lattice_size is None:
            self.init_lattice = init_lattice
            self.lattice_size = tuple(init_lattice.shape)  # type: ignore
        else:
            raise RuntimeError("init_lattice or lattice_size must be provided")

        self.eq_steps = eq_steps
        self.sim_steps = sim_steps

        self.kB = kB
        self.temp = temp

        self.beta = 1 / (self.kB * self.temp)
        self.coupl_const = coupl_const
        self.flip_frac = flip_frac

        self.curr_lattice = self.init_lattice

    def _do_mc_step(self):

        sel = np.random.rand(*self.lattice_size) < self.flip_frac

        trial_new_lattice = self.curr_lattice.copy()

        trial_new_lattice[sel] *= -1

        # probably the most expensive part of the simulation
        E_old = calc_lattice_energy(
            lattice=self.curr_lattice, coupl_const=self.coupl_const
        )
        E_new = calc_lattice_energy(
            lattice=trial_new_lattice, coupl_const=self.coupl_const
        )

        delta_E = E_new - E_old

        if delta_E < 0 or (np.random.rand() < np.exp(-self.beta * delta_E)):
            self.curr_lattice = trial_new_lattice  # Accept the new configuration

    def _equilibrate(self, n_steps: int):

        for i in range(n_steps):
            self._do_mc_step()

    def _simulate(self, n_steps: int) -> npt.NDArray[np.int32]:

        results = np.full(
            (n_steps + 1, *self.lattice_size), fill_value=0, dtype=np.int32
        )

        results[0] = self.curr_lattice

        for i in range(1, n_steps + 1):

            self._do_mc_step()

            results[i] = self.curr_lattice

        gc.collect()

        return results

    def run(self) -> IsingMCMCResult:

        start = time.time()

        self._equilibrate(n_steps=self.eq_steps)
        results = self._simulate(n_steps=self.sim_steps)

        end = time.time()

        time_taken = round(end - start)

        print(
            f"Sim complete in {timedelta(seconds=time_taken)} (size={self.lattice_size}, temp={self.temp})."
        )

        return IsingMCMCResult(
            results=results,
            lattice_size=self.lattice_size,
            temp=self.temp,
            beta=self.beta,
            coupl_const=self.coupl_const,
            kB=self.kB,
            eq_steps=self.eq_steps,
            sim_steps=self.sim_steps,
            flip_frac=self.flip_frac,
        )
