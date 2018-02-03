import numpy as np
import math
import matplotlib.pyplot as plt

C = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
test_acc = [0.7675, 0.7875, 0.7825, 0.7625, 0.7625, 0.7625, 0.7625, 0.7625, 0.7625, 0.7625]
train_acc = [0.8559, 0.9139, 0.9879, 0.9967, 0.9988, 1, 1, 1, 1, 1]
log2C = []

for i in C:
    log2C.append(math.log(i, 10))

print log2C
plt.plot(log2C, test_acc)
plt.plot(log2C, train_acc)
plt.show()
