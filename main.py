import matplotlib.pyplot as plt
from light_prop.propagation_facade import PropagationFacade
from light_prop.propagation.params import PropagationParams
from light_prop.propagation.methods import ConvolutionPropagation


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()
    out = PropagationFacade(params).progagate(ConvolutionPropagation)
    plt.imshow(out.to_abs(), interpolation='nearest')
    plt.show()

