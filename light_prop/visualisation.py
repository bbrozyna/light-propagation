import os
import logging
from pathlib import Path
from matplotlib import pyplot as plt

from light_prop.propagation_results import PropagationResult


class GeneratePropagationPlot:
    PLOT_INTENSITY = "intensity"
    PLOT_PHASE = "phase"
    PLOT_ABS = "abs"

    def __init__(self, propagation_result: PropagationResult):
        self.propagation_result = propagation_result

    def save_output_as_figure(self, path, output_type=PLOT_ABS):
        logging.info("Plotting image data")
        plot_type = {
            self.PLOT_ABS: self.propagation_result.to_abs,
            self.PLOT_INTENSITY: self.propagation_result.to_intensity,
            self.PLOT_PHASE: self.propagation_result.to_phase,
        }
        data = plot_type[output_type]()
        self._prepare_path_to_save(path)
        plt.imshow(data, interpolation='nearest')
        plt.savefig(path)
        logging.info("Generated")

    def _prepare_path_to_save(self, path):
        logging.info('Preparing directories')
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)
