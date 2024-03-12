import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import os
import librosa
from scipy import stats
import math

# setups
np.random.seed()
scaler = MaxAbsScaler()


def relu_activation(neuron_value):
    return max(0, neuron_value)


def softmax_activation(inputs):
    return np.exp(inputs) / np.sum(np.exp(inputs))


# single neuron, has single bias and a list of weights equal to the amount of inputs from the layer before
class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(1, n_inputs)
        # for initial setup set bias as 0
        self.bias = 0
        self.neuron_value = 0

    def forward(self, inputs):
        self.neuron_value = np.dot(self.weights, inputs)[0] + self.bias
        print("Neuron output: input/weights/bias/output_before_activation")
        print(inputs, self.weights, self.bias, self.neuron_value)
        return self.neuron_value


# single layer, has a set amount of neurons
class Layer:
    def __init__(self, n_inputs, n_neurons, is_input_layer, is_output_layer):
        self.neurons = []
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def forward(self, inputs):
        if self.is_input_layer:
            return inputs
        elif self.is_output_layer:
            # as probability distribution (softmax)
            return self.get_final_outputs(inputs)

        outputs = []
        for neuron in self.neurons:
            outputs.append(relu_activation(neuron.forward(inputs)))
        return outputs

    def get_final_outputs(self, inputs):
        neuron_values = []
        for neuron in self.neurons:
            neuron_values.append(neuron.forward(inputs))

        # subtract max value to avoid overflow in exp function, result doesn't change
        neuron_values = neuron_values - max(neuron_values)
        return softmax_activation(neuron_values)


# entire network of multiple layers
class Network:
    def __init__(self, n_hidden_layers, n_input_features, n_output_classes, n_hidden_nodes):
        self.layers = []

        # add input layer
        self.layers.append(Layer(0, n_input_features,is_input_layer=True, is_output_layer=False))

        # add hidden layers
        n_inputs = n_input_features
        for i in range(n_hidden_layers):
            n_neurons = n_hidden_nodes[i]
            self.layers.append(Layer(n_inputs, n_neurons, is_input_layer=False, is_output_layer=False))
            n_inputs = n_hidden_nodes[i]

        # add output layer
        self.layers.append(Layer(n_inputs, n_output_classes, is_input_layer=False, is_output_layer=True))

    def predict(self, inputs):
        print("predict on ", inputs)
        layer_result = inputs[:50]

        # genre
        result_class = inputs['genre']
        for layer in self.layers:
            layer_result = layer.forward(layer_result)

        # final result as class probabilities
        print("sum", np.sum(layer_result))
        print("Final results: ", layer_result)

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


def normalize(audio_data):
    max_abs = np.max(np.abs(audio_data))
    norm_data = audio_data / max_abs
    return norm_data


def feature_extraction(file):
    x, sample_rate = librosa.load(file)
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


if __name__ == "__main__":
    # read data
    columns = ['audio_data', 'genre']
    data = []
    genres = []

    print("Reading data files...")
    for folder, subs, files in os.walk('../archive (15)/Data/genres_original'):
        for filename in files:
            try:
                data.append([feature_extraction(folder + '/' + filename), os.path.basename(folder)])
            except IOError:
                print("Could not open ", filename)
                continue

    df = pd.DataFrame(data, columns=columns)
    df_genres = df['genre'].copy()
    df_with_features = pd.DataFrame(df['audio_data'].tolist())

    # data augmentation coming soon
    '''for column in df_with_features:
        plt.figure()
        df_with_features.boxplot([column])
        plt.show()
    '''

    df_with_features = pd.DataFrame(scaler.fit_transform(df_with_features))
    df_all_columns = pd.concat([df_with_features, df_genres], axis=1)

    # remove outliers
    df_all_columns = df_all_columns[(np.abs(stats.zscore(df_all_columns.iloc[:, :50])) < 3).all(axis=1)]

    print("Building network...")
    # build network
    layer_nodes = [10, 10]
    n_hidden_layers = len(layer_nodes)
    n_output_classes = 10
    n_inputs = 50

    nn = Network(n_hidden_layers, n_inputs, n_output_classes, layer_nodes)

    print("visualize network")
    nn.print_network()

    print("Example on a single input...")
    nn.predict(df_all_columns.iloc[0])
