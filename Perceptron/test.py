import perceptron as perc
import numpy as np
import matplotlib.pyplot as plt

p = perc.Perceptron()  # create a perceptron

def generate_gaussian_points(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    return points

# specify first cluster
mean1 = [0, 0]
cov1 = [[0.2, 0], [0, 0.2]]

# specify second cluster
mean2 = [3, 3]
cov2 = [[0.2, 0], [0, 0.2]]

num_points = 10
ones = np.ones((num_points, 1))

# Generate two clusters of points that are linearly separable
points1 = generate_gaussian_points(mean1, cov1, num_points)
points1 = np.hstack((points1, ones))       # extend points from (x1, x2) to (x1, x2, 1) (in order to be compatible with the weight vector (w1, w2, w3))

points2 = generate_gaussian_points(mean2, cov2, num_points)
points2 = np.hstack((points2, ones))

ones   = np.ones(num_points)
zeros  = np.zeros(num_points)
labels = np.concatenate((ones, zeros))

training_data = np.concatenate((points1, points2), axis=0)

p.train(training_data, labels)  # train the perceptron on the training data

# Plot evolution of the decision boundaries
n_updates = len(p.weightsequence)
x = np.linspace(-5, 5, 1000)

color = iter(plt.cm.rainbow(np.linspace(0, 1, n_updates)))
for i in range(n_updates-1):
   c = next(color)
   weights = p.weightsequence[i]
   y = (-weights[2]-weights[0] * x) / weights[1]
   if i == 0:
       plt.plot(x, y, c='red', linewidth=3, label='First boundary')
   else:
       plt.plot(x, y, c=c, label=str(i+1))

y = (-p.weights[2]-p.weights[0] * x) / p.weights[1]
plt.plot(x, y, c='blue', linewidth=3, label='Last boundary')


for i in range(num_points):
    plt.scatter(points1[i][0], points1[i][1], color='black')
    plt.scatter(points2[i][0], points2[i][1], color='black')


plt.xlim(-3, 6)  # Set x-axis limits
plt.ylim(-3, 6)  # Set y-axis limits
plt.legend()
plt.title("Evolution of decision boundaries during training")
plt.xlabel("input feature 1")
plt.ylabel("input feature 2")
plt.show()
