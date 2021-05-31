from PIL import Image
from unit10 import c1w4_utils as u10
from DL2 import *


def start(n):
    print(f"-------------------- Exercise {n} --------------------")


def end():
    print("-------------------------------------------------------\n")


# -------------------- Exercise 2.0 --------------------
start(2)
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W4()

# Example of a picture
index = 87
plt.imshow(train_set_x_orig[index])
plt.show()
print(f"y = {str(train_set_y[0, index])}. It's a {classes[train_set_y[0, index]].decode('utf-8')} picture")
end()

# -------------------- Exercise 2.1 --------------------
start(2.1)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[index].shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_set_x_orig.shape))
print("train_y shape: " + str(train_set_y.shape))
print("test_x_orig shape: " + str(test_set_x_orig.shape))
print("test_y shape: " + str(test_set_y.shape))
end()

# -------------------- Exercise 2.2 --------------------
start(2.2)
train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = (train_set_x_flatten / 255) - 0.5
test_set_x = (test_set_x_flatten / 255) - 0.5

print("train_x's shape:", train_set_x.shape)
print("test_x's shape:", test_set_x.shape)
print("normalized train color: ", train_set_x[10][10])
print("normalized test color:", test_set_x[10][10])
end()

# -------------------- Exercise 2.3 --------------------
start(2.3)
model = DLModel()
model.add(DLLayer("Perceptron 1", 7, (train_set_x.shape[0],), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 2", 1, (7,), "sigmoid", "Xavier", 0.007))
model.compile("cross entropy")
costs = model.train(train_set_x, train_set_y, 2500)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_set_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_set_x) == test_set_y))
end()

# -------------------- Exercise 2.4 --------------------
start(2.4)
model = DLModel()
model.add(DLLayer("Perceptron 1", 30, (train_set_x.shape[0],), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 2", 15, (30,), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 3", 10, (15,), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 4", 10, (10,), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 5", 5, (10,), W_initialization="Xavier", learning_rate=0.007))
model.add(DLLayer("Perceptron 6", 1, (5,), "sigmoid", "Xavier", learning_rate=0.007))

model.compile("cross entropy")
costs = model.train(train_set_x, train_set_y, 2500)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_set_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_set_x) == test_set_y))
end()

# -------------------- Exercise 2.5 --------------------
start(2.5)
img_path = r'cat.jpg'  # full path of the image
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
img = Image.open(img_path)
image64 = img.resize((num_px, num_px), Image.ANTIALIAS)
plt.imshow(img)
plt.show()
plt.imshow(image64)
plt.show()
my_image = np.reshape(image64, (num_px * num_px * 3, 1))
my_image = my_image / 255. - 0.5
p = model.predict(my_image)
print("L-layer model predicts a \"" + classes[int(p),].decode("utf-8") + "\" picture.")
end()
