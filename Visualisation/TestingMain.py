"""
This python program is part of degree Project in Vehicle Engineering, and 
is an test of stepping schedules for a NN. Generated data is saved as .dat 
files in folder "results". 

3 neccessary files; TestingMain.py, TestingNetwork.py, TestingHelpfuncs.py

TestingMain will run train and test the network with given stepping schedule, 
change parameter schemeVariant to try different schedules

TestingNetwork has global variables which can be modified to adjust network 
architecture and training parameters such as used share of data, batch size

TestingHelpfuncs contains various functions necessary to run the program

Authors:    Axel Boivie <aboivie@kth.se>
            Victor Bellander <vicbel@kth.se>
Course:     SA115X Degree Project in Vehicle Engineering, 
            First Level, 15.0 Credits, KTH
Project:    Implementation and Optimisation of a Neural Network
            Spring 2021, KTH, Stockholm 
GitHub:     https://github.com/axeboii/NeuralNetwork_SA115X
Last modified 2021/04/19 by AB
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from TestingNetwork import Network
from TestingHelpfuncs import Helpfuncs as hf
#from keras.datasets import fashion_mnist as mnist
from keras.datasets import mnist

def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    nameList = ['trainLoss', 'testLoss', 'trainAcc', 'testAcc']
    steppingSchedules = ["inverseTimeDecay", "inverseTimeDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay", "ADAM"]
    learningRates = [0.5, 0.5, 0.5, 0.3, 0.3, 0.003]
    decayRates = [0.1, 0.5, 2, 0, 2.5, 0]
    # Loop through stepping schedule
    for i, steppingSchedule in enumerate(steppingSchedules):
        trainLossMtrx = []
        testLossMtrx = []
        trainAccMtrx = []
        testAccMtrx = []
        learningRate = learningRates[i]
        decayRate = decayRates[i]
        # Each scheme is run 10 times
        for i in range(1, 11):
            NN = Network()
            print("-"*50)
            print('Körning ', i)
            trainAccVec, trainLossVec, testAccVec, testLossVec = NN.train(train_X, train_y, test_X, test_y, steppingSchedule, learningRate, decayRate)
            trainLossMtrx.append(trainLossVec)
            testLossMtrx.append(testLossVec)
            trainAccMtrx.append(trainAccVec)
            testAccMtrx.append(testAccVec)
        # Write data to file
        nameEnd = '_' + steppingSchedule + '_' + str(learningRate) + '_' + str(decayRate) 
        hf.writeToFile(trainLossMtrx, 'TrainLoss' + nameEnd)
        hf.writeToFile(testLossMtrx, 'TestLoss' + nameEnd)
        hf.writeToFile(trainAccMtrx, 'TrainAcc' + nameEnd)
        hf.writeToFile(testAccMtrx, 'TestAcc' + nameEnd)

if __name__ == "__main__":
    main()
       


