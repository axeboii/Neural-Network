"""
This python program is part of degree Project in Vehicle Engineering, 
and is an implementation of a fully connected neural network

3 neccessary files; main.py, Network.py, Helpfuncs.py

main will run train and test the network with given stepping schedule, 
change parameter schemeVariant to try different schedules

Network has global variables which can be modified to adjust network architecture and 
training parameters such as used share of data, batch size

Helpfuncs contains various functions necessary to run the program

Authors:    Axel Boivie <aboivie@kth.se>
            Victor Bellander <vicbel@kth.se>
Course:     SA115X Degree Project in Vehicle Engineering, 
            First Level, 15.0 Credits, KTH
Project:    Implementation and Optimisation of a Neural Network
            Spring 2021, KTH, Stockholm 
GitHub:     https://github.com/axeboii/NeuralNetwork_SA115X
Last modified 2021/05/23 by AB
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Network import Network
from Helpfuncs import Helpfuncs as hf
#from keras.datasets import fashion_mnist as mnist
from keras.datasets import mnist

def main():
    # Load and adapt data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    # Choose stepping scheme
    schemes = ["No stepping scheme", "exponentialDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay", "ADAM"]
    learningRates = [0.3, 0.5, 0.5, 0.3, 0.3, 0.003]
    decayRates = [0, 0.75, 0.5, 0, 2.5, 0]
    # Pick scheme 0-5 for 0: No scheme, 1: ed, 2: itd, 3: pcd, 4: pd, 5: ADAM
    schemeVariant = 5
    # Create the network
    NN = Network()
    # Train the network
    NN.train(train_X, train_y, test_X, test_y, schemes[schemeVariant], learningRates[schemeVariant], decayRates[schemeVariant])
    # Test the network
    hf.test(NN, test_X, test_y)

if __name__ == '__main__':
    main()