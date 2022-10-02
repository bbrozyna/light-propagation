import logging
import os
from pathlib import Path

from matplotlib import pyplot as plt

from lightprop.lightfield import LightField


# TODO enum?
class PlotTypes:
    INTENSITY = "intensity"
    PHASE = "phase"
    ABS = "abs"


class Plotter:
    def __init__(self, propagation_result: LightField, output_type=PlotTypes.ABS):
        self.propagation_result = propagation_result
        logging.info("Plotting image data")
        plot_type = {
            PlotTypes.ABS: self.propagation_result.get_abs,
            PlotTypes.INTENSITY: self.propagation_result.to_intensity,
            PlotTypes.PHASE: self.propagation_result.get_phase,
        }
        self.data = plot_type[output_type]()

        plt.imshow(self.data, interpolation="nearest")

    def save_output_as_figure(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.savefig(path)
        logging.info("Generated")

    def show(self):
        plt.show()

    def _prepare_path_to_save(self, path):
        logging.info("Preparing directories")
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)
