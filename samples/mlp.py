## This is course material for Introduction to Modern Artificial Intelligence
## Example code: mlp.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# Load dependencies
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Create data
linearSeparableFlag = False
x_bias = 6

def toy_2D_samples(x_bias ,linearSeparableFlag):
    label1 = np.array([[1, 0]])
    label2 = np.array([[0, 1]])

    if linearSeparableFlag:

        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        """
        np.random.multivariate_normal() 
        1st arg --> mean vector: the point around which the other points will center
        2nd arg --> covariance matrix: defines how spread out the data points are
        - 2x2 matrix bc we have to dimensions, x and y
        - The 1s in our example mean the data is spread out around the mean w/ a std deviation of 1 
        - The 0s in our example mean there is no correlation btw the dimensions
         --> i.e. the spread of points in the x-direction does not affect the spread in the y-direction
        3rd arg --> number of samples/size

        Thus each sample here is a 2D array w/ 100 rows and 2 columns (for the 2 dimensions, x & y)
        """

        samples = np.concatenate((samples1, samples2 ), axis =0) # so here it would have 200 rows and 2 columns
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 50)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 50)

        samples = np.concatenate((samples1, samples2, samples3, samples4 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()

    label1 = np.array([[1, 0]]) # labels for each class
    label2 = np.array([[0, 1]])
    labels1 = np.repeat(label1, 100, axis = 0) # makes 100 rows of label1 --> [1, 0]
    labels2 = np.repeat(label2, 100, axis = 0)
    labels = np.concatenate((labels1, labels2 ), axis =0) # puts together array w/ 200 rows, each w/ one label
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set
randomOrder = np.random.permutation(200) # generates a 1D array w/ random permutations of integers 0–199
# E.g. could be [ 23, 45, 89, 10, 5, ..., 193, 178]
trainingX = samples[randomOrder[0:100], :] # randomly splits some samples & correct labels as "training" data
trainingY = labels[randomOrder[0:100], :]
testingX = samples[randomOrder[100:200], :] # witholds the rest for "testing" on
testingY = labels[randomOrder[100:200], :]

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid', use_bias=True))
# model.add(Dense(4, input_shape=(2,), activation='relu', use_bias=True))
model.add(Dense(2, activation='softmax' ))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model.fit(trainingX, trainingY, epochs=500, batch_size=10, verbose=1, validation_split=0.2)

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
for i in range(100):
    predict_x=model.predict(np.array([testingX[i,:]])) 
    estimate=np.argmax(predict_x,axis=1) 
    # Gets the index of the highest probability in the prediction, which corresponds to the predicted class.

    if testingY[i,estimate] == 1:
        score = score  + 1

    if estimate == 0:
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

print('Test accuracy:', score/100)
plt.show()
