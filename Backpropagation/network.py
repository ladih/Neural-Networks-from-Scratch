import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_sizes):
        # network structure
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_sizes = hidden_sizes
        self.num_layers   = len(self.hidden_sizes)

        # initialize weights. '_ext' for weight matrices including weights for bias unit
        self.weights_ext = []
        self.weights_ext.append(np.random.randn(input_size + 1, hidden_sizes[0]))

        for i in range(self.num_layers - 1):
            self.weights_ext.append(np.random.randn(hidden_sizes[i] + 1, hidden_sizes[i+1]))

        self.weights_ext.append(np.random.randn(hidden_sizes[-1] + 1, output_size))

        self.weights = [np.delete(w, -1, axis=0) for w in self.weights_ext]  # weights without biases

        self.errors = [] # List to store training errors during the training process

    def train(self, training_inputs, labels, learning_rate, epochs, batch_size, error_threshold=1e-2):

        self.errors.append(self.total_error(training_inputs, labels))

        for epoch in range(epochs):
            dW_tot = [np.zeros(w.shape) for w in self.weights_ext]

            batch_in, batch_out = self._get_batch(training_inputs, labels, batch_size)

            # forward step
            for inp, label in zip(batch_in, batch_out):
                out_layer_ext = np.append(inp, 1)  # initialize out_layer_ext with input in extended form
                out_layers_ext = [out_layer_ext]   # initialize out_layers_ext with extended input
                for w in self.weights_ext:
                    out_layer = sigmoid(out_layer_ext @ w)
                    out_layer_ext = np.append(out_layer, 1)
                    out_layers_ext.append(out_layer_ext)


                out_layers = [np.delete(o, -1, axis=0) for o in out_layers_ext]  # weights without biases
                out_layers_ext = [o.reshape(1, -1) for o in out_layers_ext]

                derivatives = [np.diag(o * (1 - o)) for o in out_layers] # stored derivatives in matrix form at each layer

                derivative_error = (out_layers[-1] - label).reshape(-1, 1) # assuming error at each output unit to be 1/2 * (out - label)^2

                # backpropagation step
                back_prop = derivatives[-1] @ derivative_error # propagate from error units to output units
                dW = (-learning_rate * back_prop @ out_layers_ext[-2]).T # weight changes for weights from last hidden to output units
                dW_vec = [dW]
                for k in range(self.num_layers):
                    back_prop = derivatives[-k-2] @ self.weights[-k-1] @ back_prop
                    dW = (-learning_rate * back_prop @ out_layers_ext[-k-3]).T
                    dW_vec.append(dW)

                dW_vec = dW_vec[::-1]   # dW_vec was built during backpropagation, so need to reverse it to be compatible with dW_tot

                # accumulate weight changes
                for i in range(len(dW_tot)):
                    dW_tot[i] += dW_vec[i]

            # update the weights
            for i in range(len(self.weights_ext)):
                self.weights_ext[i] += dW_tot[i]
            self.weights = [np.delete(w, -1, axis=0) for w in self.weights_ext]

            error = self.total_error(training_inputs, labels)
            self.errors.append(error)

            if error < error_threshold:
                self.print_training(epoch, learning_rate, batch_size, error_threshold, self.hidden_sizes)
                break

            if (epoch + 1) % 10000 == 0:
                print("Processed", str(epoch+1) + '/' + str(epochs))

    def _get_batch(self, training_inputs, labels, batch_size):
        """Returns random subsets of training_inputs and labels of size batch_size."""

        indices = np.random.choice(len(training_inputs), batch_size, replace=False)
        batch_inputs = np.array(training_inputs[indices])
        batch_labels = np.array(labels[indices])
        return batch_inputs, batch_labels

    def print_training(self, epoch, learning_rate, batch_size, error_threshold, hidden_sizes):
        """Just a print when training is done."""

        formatted_threshold = "{:.0e}".format(error_threshold).replace("e-0", "e-")
        print("Training stopped with error <", formatted_threshold, "after", epoch + 1, "epochs.", end="\n")
        print("Learning rate:", learning_rate)
        print("Batch size:", batch_size)
        print("Hidden_sizes:", hidden_sizes)

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
        return sum/len(training)

    def plot_error_curve(self):
        epochs = len(self.errors)
        plt.plot(range(1, epochs+1), self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training Error Curve')
        plt.show()
