import numpy as np

array = np.array([[[0, 1, 2], [10, 11, 12], [20, 21, 22]], [[100, 101, 102], [110, 111, 112], [120, 121, 122]]])

flat_array = array.flatten('F')
reshape_array = array.reshape(-1)

print("flat shape:", flat_array)
print("reshape array:", reshape_array)
