import matplotlib.pyplot as plt
from lightprop.propagation_facade import PropagationFacade
from lightprop.propagation.params import PropagationParams
from lightprop.propagation.methods import ConvolutionPropagation, NNPropagation
from lightprop.visualisation import GeneratePropagationPlot


if __name__ == "__main__":
    # Define parameters for propagation
    params = PropagationParams.get_example_propagation_data()

    # Propagate using neural network
    outNN = PropagationFacade(params).progagate(NNPropagation)

    # Propagate using convolution method
    outConv = PropagationFacade(params).progagate(ConvolutionPropagation)

    # Plot the results
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(outConv.to_abs(), interpolation='nearest')
    axes[1].imshow(outNN.to_abs(), interpolation='nearest')
    plt.show()
    