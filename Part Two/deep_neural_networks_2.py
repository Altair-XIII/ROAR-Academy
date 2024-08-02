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

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
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

    label1 = np.array([[1, 1]])
    label2 = np.array([[-1, -1]])
    label3 = np.array([[-1, 1]])
    label4 = np.array([[1, -1]])
    labels1 = np.repeat(label1, 50, axis = 0)
    labels2 = np.repeat(label2, 50, axis = 0)
    labels3 = np.repeat(label3, 50, axis = 0)
    labels4 = np.repeat(label4, 50, axis = 0)
    labels = np.concatenate((labels1, labels2, labels3, labels4), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set

randomOrder = np.random.permutation(200)
trainingX = samples[randomOrder[0:100], :]
trainingY = labels[randomOrder[0:100], :]
testingX = samples[randomOrder[100:200], :]
testingY = labels[randomOrder[100:200], :]

model = Sequential()
# Must change the activation function to linear --> since sigmoid or ReLU will give 0 or 1 (non negative) outputs
model.add(Dense(2, input_shape=(2,), activation='linear', use_bias=True)) # accuracy 1.0
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model.fit(trainingX, trainingY, epochs=500, batch_size=10, verbose=1, validation_split=0.2)
# original: epochs = 700, batch_size = 10
# increasing epochs to 700 and batch_size to 50 gave 0.75 accuracy
# increasing epochs to 700 and batch_size to 350 gave 0.75 accuracy

# changing to relu
# decreasing batch_size to 5 gave 0.75 accuracy
# increasing batch_size to 20 gave 0.75 accuracy
# increasing epochs to 700 gave 0.75 accuracy
# increasing epochs to 700 and batch_size to 50 gave 0.75 accuracy

score = 0
for i in range(100):
    # Get prediction of data and signs of prediciton
    prediction = model.predict(np.array([testingX[i,:]])) # bc we are using linear activation function --> these will be floating point values
    print(prediction)
    predicted_signs = [np.sign(prediction[0][0]), np.sign(prediction[0][1])] # np.sign returns -1 for negative values, 0 for positive values
    print(predicted_signs)

    # Use testingY (labels) to check if predicted_signs are right
    # If the signs are the same (1 == 1 or -1 == -1) --> should be blue
    if testingY[i, 0] == testingY[i, 1] and predicted_signs[0] == predicted_signs[1]: # blue
        score = score  + 1
    # If the signs are different (1 != -1 or -1 != 1) --> should be red
    elif testingY[i, 0] != testingY[i, 1] and predicted_signs[0] != predicted_signs[1]: # red
        score = score  + 1

    # Plotting graph (blue if the signs are the same, red if not)
    if predicted_signs[0] == predicted_signs[1]:
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

print("accuracy: ", score/100)
plt.show()
