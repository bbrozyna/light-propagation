import os
import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class GeneratePropagationPlot:
    def __init__(self, propagation_strategy):
        self.propagation_strategy = propagation_strategy
        self.output = None

    def __get_output(self):
        if not self.output:
            self.output = self.get_propagation_output()
        return self.output

    def get_propagation_output(self):
        return self.propagation_strategy.propagate()

    def save_output_abs_figure(self, path):
        logging.info("Plotting image data")
        data = np.absolute(self.__get_output())
        self._prepare_path_to_save(path)
        plt.imshow(data, interpolation='nearest')
        plt.savefig(path)

    def _prepare_path_to_save(self, path):
        logging.info('Preparing directories')
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)
