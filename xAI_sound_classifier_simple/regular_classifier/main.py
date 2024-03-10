import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# setups
np.random.seed()


# single neuron, has single bias and a list of weights equal to the amount of inputs from the layer before
class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(1, n_inputs)
        # for initial setup set bias as 0
        self.bias = 0
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs)[0] + self.bias

        print("Neuron output: input/weights/bias/output")
        print(inputs, self.weights, self.bias, self.output)
        return self.output


# single layer, has a set amount of neurons
class Layer:
    def __init__(self, n_inputs, n_neurons, is_input_layer):
        self.neurons = []
        self.is_input_layer = is_input_layer
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def forward(self, inputs):
        if self.is_input_layer:
            return inputs

        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs


# entire network of multiple layers
class Network:
    def __init__(self, n_hidden_layers, n_input_features, n_output_classes, n_hidden_nodes):
        self.layers = []

        # add input layer
        self.layers.append(Layer(0, n_input_features, True))

        # add hidden layers
        n_inputs = n_input_features
        for i in range(n_hidden_layers):
            n_neurons = n_hidden_nodes[i]
            self.layers.append(Layer(n_inputs, n_neurons, False))
            n_inputs = n_hidden_nodes[i]

        # add output layer
        self.layers.append(Layer(n_inputs, n_output_classes, False))

    def train_network(self, inputs):
        test = inputs[0]
        for layer in self.layers:
            test = layer.forward(test)

        print("Final results: ", test)

    def print_network(self):
        G = nx.Graph()
        layer_count = 0

        prev_layer = None

        for layer in self.layers:
            layer_count += 1
            node_count = 0
            for neuron in layer.neurons:
                node_count += 1
                G.add_node(neuron, pos=(layer_count, node_count))
                if not layer.is_input_layer:
                    plt.text(layer_count, node_count + 0.3, neuron.bias, horizontalalignment='center')
                    for index, prev_neuron in enumerate(prev_layer.neurons):
                        G.add_edge(prev_neuron, neuron, weight=neuron.weights[0][index])

            prev_layer = layer

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=3)
        plt.show()


if __name__ == "__main__":
    # jede Zeile hier wäre ein eigenes Audio file...
    # pass in batches, but not too many, because of overfitting
    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8],
         [1, 2, 3, 4]]

    layer_nodes = [3, 5, 4, 9, 2]
    n_hidden_layers = 5
    n_output_classes = 2

    df = pd.DataFrame(X)
    # normalize data
    scaler = MaxAbsScaler()
    df = scaler.fit_transform(df)

    # how many layers?
    # adjust number of input features/classes/...
    nn = Network(n_hidden_layers, df.shape[1], n_output_classes, layer_nodes)
    nn.print_network()

    nn.train_network(X)



