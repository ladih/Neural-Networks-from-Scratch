# Testing the perceptron on two separable clusters in the plane

import perceptron as perc
import numpy as np
import matplotlib.pyplot as plt

# Initialize a perceptron
p = perc.Perceptron()

def generate_gaussian_points(mean, cov, num_points):
    points = np.random.multivariate_normal(mean, cov, num_points)
    return points

# Mean and covariance of first cluster of training samples
mean1 = [0, 0]
cov1 = [[0.2, 0], [0, 0.2]]

# Mean and covariance of second of training samples
mean2 = [3, 3]
cov2 = [[0.2, 0], [0, 0.2]]

# Number of points in each cluster
num_points = 10

# Ones for input format (x1, x2, 1)
ones = np.ones((num_points, 1))

# Generate two clusters of points that are (hopefully) linearly separable
points1 = generate_gaussian_points(mean1, cov1, num_points)
points1 = np.hstack((points1, ones)) # (x1, x2) -> (x1, x2, 1)

points2 = generate_gaussian_points(mean2, cov2, num_points)
points2 = np.hstack((points2, ones))


# Labels for training data
ones   = np.ones(num_points)
zeros  = np.zeros(num_points)
labels = np.concatenate((ones, zeros))

# Label the clusters
training_data = np.concatenate((points1, points2), axis=0)

# Train the perceptron
p.train(training_data, labels)

# Plot evolution of decision boundaries
n_updates = len(p.weightsequence)
x = np.linspace(-5, 5, 100)

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
plt.plot(x, y, c='blue', linewidth=3, label='Final boundary')


for i in range(num_points):
    plt.scatter(points1[i][0], points1[i][1], color='black')
    plt.scatter(points2[i][0], points2[i][1], color='black')


plt.xlim(-3, 6)  # Set x-axis limits
plt.ylim(-3, 6)  # Set y-axis limits
plt.legend()
plt.title("Evolution of decision boundaries during training")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


