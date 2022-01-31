import numpy as np
import scipy
import math

a = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, ],
              [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]])

b = np.array([[1.73205081, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1.73205081, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1.73205081, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1.73205081, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1.73205081, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1.73205081, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1.73205081, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.73205081]])

k = np.array([3, 3, 3, 3, 3, 3, 4, 3, 4, 3])
group1 = [1, 2, 3, 4]
group2 = [6, 7, 8, 9, 0, 5]

value = 0

# b = np.zeros([10, 10])
#
# for i in range(0, np.shape(a)[0]):
#     for j in range(0, np.shape(a)[0]):
#         if i == j:
#             b[i, j] = math.sqrt(k[i])

for i in range(0, np.shape(a)[0]):
    for j in range(0, np.shape(a)[0]):
        dummy = a[i, j] - ((k[i] * k[j]) / 36)
        if i in group1 and j in group1:
            value += dummy
        elif i in group2 and j in group2:
            value += dummy

print(value)

# k = np.array([3,3,3,3,3,3,4,3,4,3])
# group1 = [0,1,9,7,8]
# group2 = [2,3,4,5,6]
# ins = 0.7
# outs = 0.3

# a = np.zeros([10,10])
#
#
# for i in range(0, np.shape(a)[0]):
#     for j in range(0, np.shape(a)[0]):
#         if i < j:
#             if i in group1 and j in group1:
#                 a[i, j] = np.random.poisson(k[i]*k[j]*0.7, 1)
#             if i in group2 and j in group2:
#                 a[i, j] = np.random.poisson(k[i]*k[j]*0.7, 1)
#             else:
#                 a[i, j] = np.random.poisson(k[i]*k[j]*0.3, 1)
#         if j < i:
#             a[i, j] = a[j, i]

# b = np.array([[0, 2, 6, 3, 3, 3, 5, 1, 2, 1,],
#  [2, 0, 8, 1, 7, 2, 5, 0, 5, 3,],
#  [6, 8, 0, 8, 6, 2, 4, 4, 4, 3,],
#  [3, 1, 8, 0, 8, 1, 5, 2, 3, 2,],
#  [3, 7, 6, 8, 0, 3, 3, 3, 5, 5,],
#  [3, 2, 2, 1, 3, 0, 2, 3, 2, 2,],
#  [5, 5, 4, 5, 3, 2, 0, 1, 8, 3,],
#  [1, 0, 4, 2, 3, 3, 1, 0, 3, 4,],
#  [2, 5, 4, 3, 5, 2, 8, 3, 0, 1,],
#  [1, 3, 3, 2, 5, 2, 3, 4, 1, 0,]])
#
# c = np.array([[0, 6, 2, 5, 2, 3, 3, 1, 3, 4,],
#  [6, 0, 3, 1, 2, 1, 5, 4, 4, 1,],
#  [2, 3, 0, 6, 4, 6, 6, 6, 8, 4,],
#  [5, 1, 6, 0, 4, 3, 6, 3, 4, 2,],
#  [2, 2, 4, 4, 0, 5, 2, 2, 2, 1,],
#  [3, 1, 6, 3, 5, 0, 6, 2, 2, 1,],
#  [3, 5, 6, 6, 2, 6, 0, 5, 5, 3,],
#  [1, 4, 6, 3, 2, 2, 5, 0, 4, 0,],
#  [3, 4, 8, 4, 2, 2, 5, 4, 0, 0,],
#  [4, 1, 4, 2, 1, 1, 3, 0, 0, 0,]])
#
# value = 1
#
# variable = c
#
# for i in range(0, np.shape(variable)[0]):
#     for j in range(0, np.shape(variable)[0]):
#         if i < j:
#             if i in group1 and j in group1:
#                 value = value * (((k[i]*k[j]*0.7)**variable[i, j]) / math.factorial(variable[i, j])) * (math.exp(-1*k[i]*k[j]*0.7))
#             if i in group2 and j in group2:
#                 value = value * (((k[i]*k[j]*0.7)**variable[i, j]) / math.factorial(variable[i, j])) * (math.exp(-1*k[i]*k[j]*0.7))
#             else:
#                 value = value * (((k[i] * k[j] * 0.3) ** variable[i, j]) / math.factorial(variable[i, j])) * (math.exp(-1 * k[i] * k[j] * 0.3))
#
# print(value)

f = np.matmul(b, a)
print(f)
l = np.matmul(f, b)
print(l)

eigen_values, eigen_vectors = np.linalg.eig(l)

print(eigen_values)
print(eigen_vectors)
