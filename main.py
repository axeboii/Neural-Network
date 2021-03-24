import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Network import Network
import numpy as np
import matplotlib.pyplot as plt 
from Helpfuncs import Helpfuncs as hf
#from keras.datasets import fashion_mnist as mnist
from keras.datasets import mnist

"""
# Virtuellt
from input_data import read_data_sets
mnist = read_data_sets('mnist')
train_X = mnist.train.images
train_y = mnist.train.labels
test_X = mnist.test.images
test_y = mnist.test.labels
"""


def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    schemes = ["No stepping scheme", "exponentialDecay", "inverseTimeDecay"]#, "piecewiseConstantDecay", "polynomialDecay"]
    schemes2 = ["inverseTimeDecay", "inverseTimeDecay", "inverseTimeDecay"]
    schemes2 = ["exponentialDecay", "inverseTimeDecay"]#, "exponentialDecay", "exponentialDecay"]
    schemes2 = ["no", "no"]
    # bäst för SGD(ReLu): 0.3 (0,9127)
    # bäst för ADAM(ReLu): 0.003 (0,9318)
    # bäst för ED(ReLu): lr 0.1, dr 0.5 ()
    # bäst för ITD(ReLu): lr 0.1, dr 0.005 (0,9164)
    learningRates = [0.1, 0.1, 0.1]
    decayRates = [0.99, 0.005, 0.99]
    
    mtrx = []
    for i, steppingSchedule in enumerate(schemes2):
        testAccuracies = []
        """for dr in decayRates[i]:
            for lr in learningRates[i]:"""
        NN = Network()
        loss = NN.train(train_X, train_y, test_X, test_y, steppingSchedule, learningRates[i], decayRates[i])
        acc = hf.test(NN, 10000, test_X, test_y)
        testAccuracies.append(acc)
        print("----------------------------------------\n\n")
        mtrx.append(testAccuracies)
        xaxis = np.divide(list(range(1, len(loss)+1)),len(loss)/10)
        plt.plot(xaxis, loss, label = steppingSchedule)  
        # NN.writeLossFunc()
    #print("\n\n")
    for i, vec in enumerate(mtrx):
        print("Stepping schedule: ", schemes[i])
        for acc in vec:
            print(str(acc).replace(".", ","))
    
    #print("Testing accuracies: ", testAccuracies)
    plt.xlim([0, 10])
    #plt.ylim([0, 1.2])

    plt.xlabel('Epoch') 
    plt.ylabel('Loss function value') 
    plt.title('Loss function') 
    plt.legend()
    plt.show() 
    
if __name__ == '__main__':
    main()