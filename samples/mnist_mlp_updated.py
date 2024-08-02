from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 40

# input image dimensions
img_rows, img_cols = 28, 28

# load the data built in Keras, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train and y_train (and the tests) are 3D arrays, with a number of images that are 28 x 28 pixels each
# --> so (number_of_train_samples, 28, 28)

# Display some examples from first 20 images
print(y_train[0:20])
plt.figure(1)
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()

# Convert images to vectors using the reshape() function
# Specifically changes the 3D array into an array of 1D arrasy since neural networks expect 1D arrays
"""
[
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]],

  [[9, 8, 7],
   [6, 5, 4],
   [3, 2, 1]]
]
--> 
[
  [1, 2, 3, 4, 5, 6, 7, 8, 9],
  [9, 8, 7, 6, 5, 4, 3, 2, 1]
]
"""
x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols) 
x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)
input_shape = (img_rows*img_cols,)

# When calculating image data, convert from uint8 to float
# This conversion is done bc floating-point arithmetic is more precise for neural network computations.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Reduce the element range from [0, 255] to [0, 1]
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
"""
to_categorical converts class labels into a binary matrix. 
For example, if there are 10 classes (digits 0-9), a label like 3 would be converted into a vector 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], where the 1 is at the index corresponding to the class label.
"""
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(800, input_shape=input_shape, activation='relu')) # (28 * 28 + 1)*800 coefficients
# Each neuron gives one  output --> becomes the next input, bc output of last layer is the input of next layer
model.add(Dense(800, activation='relu')) # (800 + 1)*800
model.add(Dense(128, activation='relu')) # (800 + 1)* 128
model.add(Dense(num_classes, activation='softmax' )) # (128 + 1)*10

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)

print(model.summary())
print('Test loss:', score[0])
print('Test accuracy:', score[1])
