from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .properties import calc_lattice_energy, calc_lattice_magnetisation


class MonteCarloResult(ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        pass


@dataclass(frozen=True)
class IsingMCMCResult(MonteCarloResult):

    results: npt.NDArray[np.int32]
    lattice_size: Tuple[int, int]
    temp: float
    beta: float
    coupl_const: float
    kB: float
    eq_steps: int
    sim_steps: int
    flip_frac: float

    @property
    def params(self) -> Dict[str, Any]:

        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "results"
        }

    def __post_init__(
        self,
    ) -> None:
        # works around frozen dataclass
        object.__setattr__(
            self,
            "energy",
            [
                calc_lattice_energy(lattice=lattice, coupl_const=self.coupl_const)
                for lattice in self.results
            ],
        )
        object.__setattr__(
            self,
            "magnetisation",
            [calc_lattice_magnetisation(lattice=lattice) for lattice in self.results],
        )

    def show_lattice(
        self,
        step: int,
        ax: Optional[Axes] = None,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 300,
    ) -> Tuple[Figure, Axes, Colorbar]:
        if ax is not None:
            fig = ax.get_figure()
            assert isinstance(fig, Figure)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(self.results[step], cmap=cmap, vmin=-1, vmax=1, origin="lower")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{5}%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")

        return fig, ax, cbar

    def animate_lattice(
        self,
        filename: str,
        fps: int = 200,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 200,  # animation scales badly with dpi
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(self.results[0], cmap=cmap, vmin=-1, vmax=1, origin="lower")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{5}%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")
        title = ax.set_title(f"Step 0")

        def update(step):
            im.set_array(self.results[step + 1])
            title.set_text(f"Step {step + 1}")

            return im, title

        anim = FuncAnimation(fig, update, frames=len(self.results) - 1, blit=True)

        anim.save(
            f"{filename}.gif",
            # writer="ffmpeg",  # may need to install this and tell matplotlib where the writer path is
            fps=fps,
        )
