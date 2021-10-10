import numpy as np
import unit10.c1w2_utils as u10
from PIL import Image
import matplotlib.pyplot as plt
from perceptron import Perceptron


def check_cat(path, indentifier):
    img = Image.open(path)
    img = img.resize((64, 64), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    my_image = np.array(img).reshape(1, -1).T

    my_predicted_image = indentifier.predict(my_image)
    print(
        f'y = {str(np.squeeze(my_predicted_image))}, your algorithm predicts a "{classes[int(np.squeeze(my_predicted_image))].decode("utf-8")}" picture.'
    )


train_X, train_set_y, test_X, test_set_y, classes = u10.load_datasetC1W2()

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

train_X_flatten = train_X.reshape(train_X.shape[0], -1).T
test_X_flatten = test_X.reshape(test_set_y.shape[0], -1).T

train_set_x = train_X_flatten / 255.0
test_set_x = test_X_flatten / 255.0

cat_identifier = Perceptron(train_set_x, train_set_y, num_iterations=4000, learning_rate=0.005)

cat_identifier.train()

Y_prediction_test = cat_identifier.predict(test_set_x)
Y_prediction_train = cat_identifier.predict(train_set_x)

check_cat(r"./cat.jpg", cat_identifier)
check_cat("./cat2.jpg", cat_identifier)
check_cat("./cat3.jpg", cat_identifier)
