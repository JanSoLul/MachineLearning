import numpy as np
import matplotlib.pyplot as plt

training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)

test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

print("training_data.shape =", training_data.shape, ", test_data.shape =", test_data.shape)

img = training_data[100][1:].reshape(28, 28)

plt.imshow(img, cmap='gray')
plt.show()