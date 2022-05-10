import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt


class GeneratePropagationReport:
    def __init__(self, propagation_strategy):
        self.propropagation_strategy = propagation_strategy
        self.output = None

    def __get_output(self):
        if not self.output:
            self.output = self.get_propagation_output()
        return self.output

    def get_propagation_output(self):
        return self.propropagation_strategy.propagate()

    def save_output_abs_figure(self, path):
        data = np.absolute(self.__get_output())
        self.prepare_path_to_save(path)
        plt.imshow(data, interpolation='nearest')
        plt.savefig(path)

    def prepare_path_to_save(self, path):
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)
