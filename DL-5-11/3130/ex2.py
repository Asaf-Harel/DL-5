import unit10.c1w2_utils as u10

# Loading the data (cat/non-cat)
train_X, train_set_y, test_X, test_set_y, classes = u10.load_datasetC1W2()

train_X_flatten = train_X.reshape(train_X.shape[0], -1).transpose()
test_X_flatten = test_set_y.reshape(test_set_y.shape[0], -1).transpose()

print("train_set_x_flatten shape: " + str(train_X_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_X_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_X_flatten / 255.0
test_set_x = test_X_flatten / 255.0
