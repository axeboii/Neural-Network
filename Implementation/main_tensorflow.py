# USAGE: python3 main_tf.py keywords
# Example of keywords: 
# exponentialDecay, inverseTimeDecay, piecewiseConstantDecay, polynomialDecay 
# ADAM 
# fashion, fashion_mnist 

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
SIZES = [28*28, 32, 10]
DATA_POINTS = 60000
EPOCHS = 10
BATCH_SIZE = 16
USED_SHARE_OF_DATA = 0.01
LEARNING_RATE = 1
DECAY_RATE = 0.05
BATCHES = DATA_POINTS/BATCH_SIZE*USED_SHARE_OF_DATA # 10% of data = 375 batches

# Simulates own implementation with the following features:
# - 1 hidden layer with 32 nodes
# - Sigmoid activation
# - Glorot initialization of weights, zero initialization of biases
# - Stepping schedule exponentialDecay/inverseTimeDecay
# - SGD with 10 % of data
class Network: 
    def __init__(self, learningRate, steppingScheme = None, decayRate = 0):
        self.learningRate = learningRate
        self.decayRate = decayRate
        self.model = tf.keras.Sequential()
        self.sysargv = sys.argv[1:]
        if steppingScheme is not None:
            self.steppingScheme = steppingScheme
            self.sysargv.append(steppingScheme)
        self.getData()
        self.initializeNetwork()
        self.compileModel()
    
    def getData(self):
        # Determines which dataset to use
        print("\n---------------------------------------")
        if "fashion" in [arg.lower() for arg in self.sysargv] or "fashion_mnist" in [arg.lower() for arg in self.sysargv]:
            dataset = "fashion"
            print("Dataset: fashion_mnist", end = ", ")

        else:
            dataset = "mnist"
            print("Dataset: mnist", end = ", ")
        # Import the Fashion MNIST dataset
        if dataset == "fashion":
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # Import the MNIST dataset
        else:  
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
            class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        # Preprocess the data
        self.train_images = train_images / 255.0
        self.train_labels = train_labels
        self.test_images = test_images / 255.0
        self.test_labels = test_labels
        self.class_names = class_names

    def initializeNetwork(self):
        self.model.add(tf.keras.layers.Flatten(
                input_shape=(28, 28)
                )
        )
        for size in SIZES[1:]:
            self.model.add(
                tf.keras.layers.Dense(
                    units = size, 
                    activation='relu', 
                    # Initialize weights and biases
                    kernel_initializer = 'glorot_uniform',
                    bias_initializer='zeros'
                )
            )    
        
    def compileModel(self):
        if "adam" in [arg.lower() for arg in self.sysargv]:
            opt = "adam"
        else:
            opt = "SGD"
        print("Optimizer = ", opt)
        if opt == "SGD": 
            if "exponentialdecay" in [arg.lower() for arg in self.sysargv]:
                learningRate = self.exponentialDecay()
                print("Stepping scheme: exponentialDecay, Learning rate = ", self.learningRate, ", Decay rate = ", self.decayRate)
            elif "inversetimedecay" in [arg.lower() for arg in self.sysargv]:
                learningRate = self.inverseTimeDecay()
                print("Stepping scheme: inverseTimeDecay, Learning rate = ", self.learningRate, ", Decay rate = ", self.decayRate)
            elif "piecewiseconstantdecay" in [arg.lower() for arg in self.sysargv]:
                learningRate = self.piecewiseConstantDecay()
                print("Stepping scheme: piecewiseConstantDecay, Learning rate = ", self.learningRate)
            elif "polynomialdecay" in [arg.lower() for arg in self.sysargv]:
                learningRate = self.polynomialDecay()
                print("Stepping scheme: polynomialDecay, Learning rate = ", self.learningRate)
            else:
                learningRate = self.learningRate
                print("No stepping scheme, Learning rate = ", learningRate)
            opt = tf.keras.optimizers.SGD(learning_rate = learningRate)
        print("---------------------------------------")
        self.model.compile(
            optimizer = opt, 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits = True
            ),
            metrics = ['accuracy']
        )

    def trainModel(self):
        history = self.model.fit(
            self.train_images, 
            self.train_labels, 
            batch_size = BATCH_SIZE,
            epochs = EPOCHS, 
            steps_per_epoch = BATCHES
        )
        plt.plot(history.history['loss'], label=self.steppingScheme)
    
    def testModel(self):
        print('Test accuracy:')
        test_loss, test_acc = self.model.evaluate(
            self.test_images, 
            self.test_labels, 
            batch_size=100
        )
        return test_acc

    def exponentialDecay(self):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = float(self.learningRate), 
            decay_steps = BATCHES, 
            decay_rate = self.decayRate, 
            staircase = True
        )
    def inverseTimeDecay(self):
        return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate = float(self.learningRate), 
            decay_steps = BATCHES, 
            decay_rate = self.decayRate, 
            staircase = True
        )
    def piecewiseConstantDecay(self):
        bound = [i*375 for i in list(range(1,11))]
        vals = [float(self.learningRate)*i for i in [1, 0.909, 0.818, 0.727, 0.636, 0.545, 0.4545, 0.3636, 0.2727, 0.1818, 0.0909]]

        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries = bound, 
            values = vals,
        )
    def polynomialDecay(self):
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate = float(self.learningRate), 
            decay_steps = 9, 
            end_learning_rate=0.1, 
            power=0.5,
            cycle=False
        )

def main():
    initialLRs = [0.5, 0.5, 0.5]
    decayRates = [0.01, 0.1, 1]
    testAccs = []
    schemes = ["inverseTimeDecay", "inverseTimeDecay", "inverseTimeDecay"]#, "piecewiseConstantDecay", "polynomialDecay"]
    for i,s in enumerate(schemes):
        temp = []
        
        model = Network(initialLRs[i], s, decayRates[i])
        model.trainModel()
        acc = model.testModel()
        temp.append(round(acc,4))
        testAccs.append(temp)
    print("---------------------------------------")
    for vec in testAccs:
        for acc in vec:
            print(str(acc).replace(".", ","))
    plt.legend()
    plt.show()
    
    #print("Testing accuracies: ", testAccs)

main()