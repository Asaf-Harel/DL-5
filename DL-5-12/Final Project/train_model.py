import tensorflow as tf
from tensorflow.keras import layers, models, losses
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import utils

(x_train, y_train), (x_test, y_test) = utils.get_data()

print(x_test.shape)

x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)

CLASS_NAMES = ['malware', 'normal']

x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]]) / 255

x_val = x_train[-200:, :, :, :]
y_val = y_train[-200:]
x_train = x_train[:-200, :, :, :]
y_train = y_train[:-200]

# --------------------------------- model ---------------------------------
model = models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

print(print(model.summary()))
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val))

# model.save('./weights/model.h5')
# model.save_weights('./weights/weights.h5')

fig, axs = plt.subplots(2, 1, figsize=(15, 15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

print(model.evaluate(x_test, y_test))

y_pred = model.predict(x_test)

T5_lables = ['Malware', 'Normal']

ax = plt.subplot()

cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
print(cm)
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

# labels, title and ticks
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(T5_lables)
ax.yaxis.set_ticklabels(T5_lables)

plt.show()
