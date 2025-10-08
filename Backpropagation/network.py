import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_sizes):
        # initialize structure
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_sizes = hidden_sizes
        self.num_layers   = len(self.hidden_sizes)

        # Build weight matrices
        # Example (for intution) with two inputs and two nodes (excl. bias) at first hidden

        #  o ---- w11 ---- o
        #     \         /
        #       w12, w21
        #     /         \  
        #  o ---- w22 ---- o

        # Weights from input nodes (and bias node) to first hidden layer
        #
        #   [w11     w12]      # first input node to hidden layer
        #   [w21     w22]      # second input node to hidden layer
        #   [w31     w32]      # bias node to hidden layer


        # initialize weights (ext includes weights for bias node)
        self.weights_ext = []

        # weights from input nodes (and bias node) to first hidden layer
        self.weights_ext.append(np.random.randn(input_size + 1, hidden_sizes[0]))

        # weights from hidden -> hidden -> hidden ...  (including bias nodes)
        for i in range(self.num_layers - 1):
            self.weights_ext.append(np.random.randn(hidden_sizes[i] + 1, hidden_sizes[i+1]))

        # weights from last hidden (and bias) to output nodes
        self.weights_ext.append(np.random.randn(hidden_sizes[-1] + 1, output_size))

         # weights without biases
         # delete last row from each weight matrix in weights_ext
        self.weights = [np.delete(w, -1, axis=0) for w in self.weights_ext]

        self.errors = [] # List to store training errors

    def train(self, training_inputs, labels, learning_rate, n_batches, batch_size, error_threshold):

        self.errors.append(self.total_error(training_inputs, labels))

        for batch in range(n_batches):
            # initialize weight update matrices for the batch
            dW_tot = [np.zeros(w.shape) for w in self.weights_ext]

            # fetch batch from training samples
            batch_in, batch_out = self._get_batch(training_inputs, labels, batch_size)

            # forward step for one sample at a time
            # calculates the output for each layer
            # Example:
            # first layer gives o0 = sigmoid( (input, 1) @ W0 ) to layer 2
            # layer 2 gives o1 = sigmoid (o0, 1) @ W1) to layer 3
            # layer 3 gives o2 = sigmoid (o1 @ W3) to output layer

            for inp, label in zip(batch_in, batch_out):

                # collect outputs for all layers
                out_layer_ext = np.append(inp, 1)  # input in extended form
                out_layers_ext = [out_layer_ext]

                for w in self.weights_ext:
                    out_layer = sigmoid(out_layer_ext @ w)
                    out_layer_ext = np.append(out_layer, 1)
                    out_layers_ext.append(out_layer_ext)


                out_layers = [np.delete(ol, -1, axis=0) for ol in out_layers_ext]

                # 1D array -> matrix with 1 row
                out_layers_ext = [ol.reshape(1, -1) for ol in out_layers_ext]

                # stored derivatives in matrix form at each layer
                # f' = f * (1 - f) for f = sigmoid
                derivatives = [np.diag(ol * (1 - ol)) for ol in out_layers]

                # assuming error at each output unit to be 1/2 * (out - label)^2
                # 1D array -> matrix with 1 column

                derivative_error = (out_layers[-1] - label).reshape(-1, 1)

                # Backpropagation
                # backpropagate from error units to output units
                back_prop = derivatives[-1] @ derivative_error

                # weight updates for weights from last hidden to output units
                #
                #  last_hidden -------- w --------- out_layer ------- error
                #
                # d(error) / dw = d(error) / d out_layer *
                #                 * d(out_layer) / dw
                # d(out_layer) / dw = value_last_hidden * sigmoid'

                dW = (-learning_rate * back_prop @ out_layers_ext[-2]).T
                dW_vec = [dW]

                for k in range(self.num_layers):
                    back_prop = derivatives[-k-2] @ self.weights[-k-1] @ back_prop
                    dW = (-learning_rate * back_prop @ out_layers_ext[-k-3]).T
                    dW_vec.append(dW)

                # reverse dW_vec (was built during backprop)
                dW_vec = dW_vec[::-1]

                # add weight updates for the sample to total
                for i in range(len(dW_tot)):
                    dW_tot[i] += dW_vec[i]

            # update weights
            for i in range(len(self.weights_ext)):
                self.weights_ext[i] += dW_tot[i]
            self.weights = [np.delete(w, -1, axis=0) for w in self.weights_ext]

            error = self.total_error(training_inputs, labels)
            self.errors.append(error)

            if error < error_threshold:
                print(f"Training done with error {error:.4f}.")
                break

            if (batch + 1) % 10000 == 0:
                print("Processed", str(batch+1) + '/' + str(n_batches))
        if error >= error_threshold:
            print(f"Training done with final error {error:.4f}. Threshold {error_threshold} not reached (why?)")

    def _get_batch(self, training_inputs, labels, batch_size):
        """Returns random training batch (inputs+labels)."""

        indices = np.random.choice(len(training_inputs), batch_size, replace=False)
        batch_inputs = np.array(training_inputs[indices])
        batch_labels = np.array(labels[indices])
        return batch_inputs, batch_labels

    def predict(self, inp):
        output = np.append(inp, 1)
        for w in self.weights_ext:
            output = sigmoid(output @ w)
            output = np.append(output, 1)
        return np.delete(output, -1, axis=0)


    def total_error(self, training, labels):
        sum = 0
        for i, sample in enumerate(training):
            res = self.predict(sample)
            for j, component in enumerate(res):
                sum += 1/2 * (res[j] - labels[i][j])**2
        return sum / len(training)

    def plot_error_curve(self):
        n_batches = len(self.errors)
        plt.plot(range(1, n_batches+1), self.errors)
        plt.xlabel('Number of batches trained on')
        plt.ylabel('Error')
        plt.title('Training Error Curve')
        plt.show()


