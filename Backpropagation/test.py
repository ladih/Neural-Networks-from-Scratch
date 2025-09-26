from network import *
import random

# Train data
X = np.array([[0, 0], [0.5, 0], [1, 0], [0, 0.5], [0, 1], [0.25, 0.25], [0.5, 0.5], [1, 1]])  # input
y = np.array([[1], [1], [1], [1], [1], [0], [0], [0]])     # labels

nn = NeuralNetwork(input_size=len(X[0]), output_size=len(y[0]), hidden_sizes=[2, 2])

error_threshold = 1e-3 # training stops if
n_batches = 20000 # number of batches trained on
batch_size = 8       # must be <= len(X)
nn.train(X, y, learning_rate=0.1, n_batches=n_batches, batch_size=batch_size, error_threshold=error_threshold)

nn.plot_error_curve()

# Plot training data
# Class 1 points
plt.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], color="red", label="Class 1") # need ravel since input y to NN is 2D

# Class 0 points
plt.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], color="black", label="Class 0")

plt.legend()
plt.title("Training data")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
plt.show()



def classifications_2d(n_points, nn):
    """Shows classification of n_points random points in the plane.
        Only works if input = 2d, output = 1d"""
    xmin, xmax = -0.5, 1.5
    ymin, ymax = -0.5, 1.5

    # dummy points for legend
    plt.scatter([], [], color='red', label='Class 1')
    plt.scatter([], [], color='black', label='Class 0')

    for i in range(n_points):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        res = nn.predict(np.array([[x, y]]))
        if res > 0.5:
            plt.scatter(x, y, color='red')
        else:
            plt.scatter(x, y, color='black')
    plt.title("Example of how points in the plane are classified")
    plt.xlabel("input feature 1")
    plt.ylabel("input feature 2")
    plt.show()

# show classifications in the plane
classifications_2d(n_points = 500, nn=nn)
