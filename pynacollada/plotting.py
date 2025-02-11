import pynapple as nap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation


class Plot1DConvolution:
    """
    Class to plot an animation of convolving some 1D kernel with some Tsd array.

    Parameters
    ----------
    tsd :
        The Tsd object to convolve with the kernel.
    kernel :
        The 1D kernel to convolve with the array.
    index :
        The time index. Taken from the Tsd object if not provided.
    start :
        The index along the x-axis to start the animation. Defaults to the start of the window.
    interval :
        The interval between frames in milliseconds.
    figsize :
        The figure size.
    ylim :
        The y-axis limits.
    xlabel :
        The x-axis label.
    ylabel :
        The y-axis label.
    tsd_label :
        The legend label for the Tsd array
    kernel_label :
        The legend label for the kernel
    conv_label :
        The legend label for the convolution output
    split_kernel_yaxis :
        Whether or not to have a separate y-axis (i.e. use twinx()) for plotting the kernel. Useful if the kernel is magnitudes smaller/larger than the Tsd.
    """

    def __init__(
        self,
        tsd: nap.Tsd,
        kernel: NDArray,
        index: NDArray = None,
        start: int = 0,
        interval: float = 100,
        figsize: tuple = (10, 3),
        ylim: float = None,
        xlabel: str = "Time (s)",
        ylabel: str = "Count",
        tsd_label: str = "original array",
        kernel_label: str = "kernel",
        conv_label: str = "convolution",
        split_kernel_yaxis: bool = False,
    ):
        self.tsd = tsd
        self.kernel = kernel
        if index is None:
            self.index = tsd.index.values
        else:
            self.index = index
        self.start = start
        self.conv = tsd.convolve(kernel)
        self.conv_viz = np.zeros_like(tsd)
        self.frames = len(tsd) - start
        self.interval = interval
        if ylim is None:
            if split_kernel_yaxis:
                ymin = np.min((self.tsd.min(), self.conv.min()))
                ymax = np.max((self.tsd.max(), self.conv.max()))
            else:
                ymin = np.min((self.tsd.min(), self.conv.min(), self.kernel.min()))
                ymax = np.max((self.tsd.max(), self.conv.max(), self.kernel.max()))
            ylim = (ymin, ymax)
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tsd_label = tsd_label
        self.kernel_label = kernel_label
        self.conv_label = conv_label
        self.split_kernel_yaxis = split_kernel_yaxis
        (
            self.fig,
            self.kernel_line,
            self.conv_line,
            self.conv_area,
            self.top_idx_line,
            self.bottom_idx_line,
        ) = self.setup(figsize)

    def setup(self, figsize):
        """
        Initialization of the plot.
        """
        # initial placement of kernel
        kernel_full = np.zeros_like(self.tsd)
        kidx, kmid = self.kernel_bounds(0)
        if np.any(kidx):
            kernel_full[kidx] = self.kernel[: len(kidx)]

        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

        ### top plot ###
        ax = axs[0]
        # this is fixed
        ax.plot(self.index, self.tsd, label=self.tsd_label)

        # initial visible convolution output and top center line
        if kmid >= 0:
            self.conv_viz[: kmid + 1] = self.conv[: kmid + 1]
            cx = self.index[kmid]
        else:
            cx = self.index[0]
        top_idx_line = ax.plot((cx, cx), self.ylim, "--", color="black", alpha=0.5)[0]

        # initial filled area
        conv_area = ax.fill_between(
            self.index,
            np.zeros_like(self.tsd),
            self.tsd * kernel_full.values,
            alpha=0.5,
            color="green",
        )

        # initial kernel plot
        if self.split_kernel_yaxis:
            ax = ax.twinx()
            ax.set_ylabel(self.kernel_label)
            ax.set_ylim((kernel_full.min(), kernel_full.max()))
        kernel_line = ax.plot(
            self.index, kernel_full, color="orange", label=self.kernel_label
        )[0]

        ### bottom plot ###
        ax = axs[1]
        # initial convolution output and bottom plot center line
        conv_line = ax.plot(
            self.index, self.conv_viz, color="green", label=self.conv_label
        )[0]
        bottom_idx_line = ax.plot((cx, cx), self.ylim, "--", color="black", alpha=0.5)[
            0
        ]

        ax.set_ylim(self.ylim)

        fig.legend()
        fig.supxlabel(self.xlabel)
        fig.supylabel(self.ylabel)
        plt.tight_layout()

        return fig, kernel_line, conv_line, conv_area, top_idx_line, bottom_idx_line

    def update(self, frame):
        if frame > 0:
            # place kernel at shifted location based on frame number
            kernel_full = np.zeros_like(self.tsd)
            kidx, kmid = self.kernel_bounds(frame)
            kernel_full[kidx] = self.kernel[: len(kidx)]
            self.kernel_line.set_ydata(kernel_full)

            # update visible convolution output
            if kmid >= 0:
                self.conv_viz[kmid] = self.conv[kmid]
                self.conv_line.set_ydata(self.conv_viz)
                self.top_idx_line.set_xdata((self.index[kmid], self.index[kmid]))
                self.bottom_idx_line.set_xdata((self.index[kmid], self.index[kmid]))

            # update filled area
            self.conv_area.set_data(
                self.index, np.zeros_like(self.tsd), self.tsd * kernel_full.values
            )

    def run(self):
        anim = FuncAnimation(
            self.fig, self.update, self.frames, interval=self.interval, repeat=True
        )
        plt.close(self.fig)
        return anim

    def kernel_bounds(self, frame):
        # kernel bounds set to the left of the frame index and start location
        kmin = frame + self.start - len(self.kernel)
        kmax = frame + self.start

        # kernel indices no less than 0 and no more than the length of the Tsd
        kidx = np.arange(np.max((kmin, 0)), np.min((kmax, len(self.tsd))))

        # convolution output w.r.t. the midpoint of where the kernel is placed
        kmid = kmin + np.floor(len(self.kernel) / 2).astype(int)

        return kidx, kmid


def animate_1d_convolution(tsd: nap.Tsd, kernel: NDArray, **kwargs):
    """
    Animate the convolution of a 1D kernel with some Tsd array.

    Parameters
    ----------
    tsd : nap.Tsd
        The Tsd object to be convolved.
    kernel : np.ndarray
        The 1D kernel to convolve with the array.
    **kwargs
        Additional keyword arguments to pass to Plot1DConvolution.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
    """
    anim = Plot1DConvolution(tsd, kernel, **kwargs)
    return anim.run()
