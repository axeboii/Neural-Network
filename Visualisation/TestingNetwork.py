import random
import numpy as np
from TestingHelpfuncs import Helpfuncs as hf
import math

# Global variables
SIZES = [28*28, 32, 10] # Number of nodes in: [input layer, hidden layer, ... , output layer]
EPOCHS = 10
BATCH_SIZE = 16
USED_SHARE_OF_DATA = 0.1
#LEARNING_RATE = 5
END_LEARNING_RATE = 0.1
WEIGHT_BOUND = math.pow(6/(SIZES[0]+SIZES[-1]), 0.5)
BIAS_BOUND = 0

# A class representing a Neural Network
class Network:
    def __init__(self):
        self.sizes = SIZES
        self.L = len(self.sizes)
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialLR = 0
        self.lr = 0
        self.dr = 0
        self.sc = ""
        self.correct = 0
        self.used = 0
        self.lossFuncValues = []
        for i in range(0,self.L-1):
            self.weights.append(np.random.uniform(low=-WEIGHT_BOUND, high=WEIGHT_BOUND, size=(self.sizes[i+1],self.sizes[i])))
            self.biases.append(np.random.uniform(low=-BIAS_BOUND, high=BIAS_BOUND, size=(self.sizes[i+1])))
    # Calculates values of each node in next layer.
    def nextLayer(self, a, layer):
        b = self.biases[layer]
        w = self.weights[layer]        
        #a1 = hf.sigmoid(np.matmul(w, a) + b)
        a1 = hf.relu(np.matmul(w, a) + b)
        return a1
    # Loops through all the layers and creates a prediction
    def predict(self, image):
        a = hf.flatten(image)
        for l in range(0,self.L-1):
            a = self.nextLayer(a, l)
        return a
    # Trains the NN
    def train(self, train_X, train_y, test_X, test_y, steppingSched = "No stepping scheme", learningRate = 1, decayRate = 0):
        self.lr = learningRate
        self.dr = decayRate
        self.sc = steppingSched
        self.initialLR = learningRate
        
        if steppingSched == "ADAM":
            print("----------------------------------------")
            print("SGD-ADAM", ", Alpha = ", self.lr)
            print("----------------------------------------")
            trainAccVec, trainLossVec, testAccVec, testLossVec = self.SGDADAM(train_X, train_y, test_X, test_y) 
        else:
            print("----------------------------------------")
            print("SGD, Stepping scheme = ",self.sc, ", Learning rate = ", self.lr, ", Decay rate = ", self.dr)
            print("----------------------------------------")
            trainAccVec, trainLossVec, testAccVec, testLossVec = self.SGD(train_X, train_y, test_X, test_y)        
        return trainAccVec, trainLossVec, testAccVec, testLossVec
    # Updates the learning rate according to a stepping schedule
    def updateLearningRate(self, epoch):
        if self.sc == "exponentialDecay":
            self.lr = hf.exponentialDecay(self.initialLR, self.dr, epoch)
        elif self.sc == "polynomialDecay":
            self.lr = hf.polynomialDecay(self.initialLR, END_LEARNING_RATE, epoch, EPOCHS)
        elif self.sc == "inverseTimeDecay":
            self.lr = hf.inverseTimeDecay(self.initialLR, self.dr, epoch)
        elif self.sc == "piecewiseConstantDecay":
            self.lr = hf.piecewiseConstantDecay(self.initialLR, epoch)
        else: 
            pass
    # Calculates the gradient of loss function based on all weights and biases
    def stochasticGradient(self, train_X, train_y, trainLossVec):
        changeWeights = [0]*(self.L-1)
        changeBiases = [0]*(self.L-1)
        lossFuncSum = 0
        for i in range(0, BATCH_SIZE):
            aVec = []
            DVec = []
            deltaVec = []
            k = random.randint(0,len(train_X)-1)
            xk = train_X[k]
            yk = train_y[k]
            a = hf.flatten(xk)
            aVec.append(a)
            # Performs back-propagation for all layers
            for l in range(0,self.L-1):
                z = np.matmul(self.weights[l],a)+self.biases[l]
                a = hf.relu(z)
                D = np.diag(hf.reluPrim(z))
                #a = hf.sigmoid(z)   
                #D = np.diag(hf.sigmoidPrim(z))
                aVec.append(a)
                DVec.append(D)
            delta_L = np.matmul(DVec[-1],(a-hf.formatY(yk)))
            deltaVec.append(delta_L)
            for l in reversed(range(-self.L+1, -1)):
                delta_l = np.matmul(DVec[l], np.matmul(np.transpose(self.weights[l+1]), deltaVec[l+1]))
                deltaVec.insert(0, delta_l)
            for l in reversed(range( -self.L+1, 0)):
                changeBiases[l] += deltaVec[l]
                changeWeights[l] += np.outer(deltaVec[l], aVec[l-1])
            prediction = max(aVec[-1])
            index = aVec[-1].index(prediction)
            if (index == int(yk)):
                self.correct += 1
            lossFuncSum += hf.lossFunc(aVec[-1], yk)
        trainLossVec.append(lossFuncSum/BATCH_SIZE)
        # Calculates average values
        dw = [cw/BATCH_SIZE for cw in changeWeights]
        db = [cb/BATCH_SIZE for cb in changeBiases]
        return dw, db
    # Trains the network based on Stochastic Gradient Descent
    def SGD(self, train_X, train_y, test_X, test_y):
        trainLossVec = []
        testAccVec = []
        testLossVec = []
        trainAccVec = []
        # Epoch loop
        for i in range(1,EPOCHS+1):
            self.correct = 0
            self.used = 0
            # Batch loop
            for j in range(1, int(len(train_X)*USED_SHARE_OF_DATA/BATCH_SIZE)+1):
                dw, db = self.stochasticGradient(train_X, train_y, trainLossVec)
                self.used += BATCH_SIZE
                """print("\n", self.weights[1][5][5], "\n")
                input()"""
                # Layer loop
                for l in range(1, self.L):
                    self.weights[l-1] -= self.lr*dw[l-1]
                    self.biases[l-1] -= self.lr*db[l-1]
                hf.progress_bar(j, int(len(train_X)*USED_SHARE_OF_DATA/BATCH_SIZE), EPOCHS, i, round(100*self.correct/(self.used), 4))
                testAcc, testLoss = hf.test(self, 50, test_X, test_y)
                testAccVec.append(testAcc)
                testLossVec.append(testLoss)
                trainAcc = self.correct/self.used
                trainAccVec.append(trainAcc) 
                   
            self.updateLearningRate(i) 
            
        #loss = hf.averageLoss(self.lossFuncValues)
        return trainAccVec, trainLossVec, testAccVec, testLossVec

    # Trains the network based on ADAM
    def SGDADAM(self, train_X, train_y, test_X, test_y):
        trainLossVec = []
        testAccVec = []
        testLossVec = []
        trainAccVec = []
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        alpha = self.lr
        t = 1
        Vdw = [0]*(self.L-1)
        Sdw = [0]*(self.L-1)
        Vdb = [0]*(self.L-1)
        Sdb = [0]*(self.L-1)
        # Epoch loop
        for i in range(1, EPOCHS+1):
            self.correct = 0
            self.used = 0
            # Minibatch loop
            for j in range(1, int(len(train_X)*USED_SHARE_OF_DATA/BATCH_SIZE+1)):
                dw, db = self.stochasticGradient(train_X, train_y, trainLossVec)
                self.used += BATCH_SIZE
                """print("\n", self.weights[1][5][5], "\n")
                input()"""
                # Layer loop
                for l in range(1, self.L):   
                    # Update first and second moments
                    Vdw[l-1] = beta1*Vdw[l-1] + (1-beta1)*dw[l-1]
                    Vdb[l-1] = beta1*Vdb[l-1] + (1-beta1)*db[l-1]
                    Sdw[l-1] = beta2*Sdw[l-1] + (1-beta2)*(np.square(dw[l-1]))
                    Sdb[l-1] = beta2*Sdb[l-1] + (1-beta2)*(np.square(db[l-1]))
                    # Get corrected values
                    Vdwcor = Vdw[l-1]/(1-beta1**t)
                    Vdbcor = Vdb[l-1]/(1-beta1**t)
                    Sdwcor = Sdw[l-1]/(1-beta2**t)
                    Sdbcor = Sdb[l-1]/(1-beta2**t)
                    # Update weights and biases
                    cw = np.divide(Vdwcor, np.sqrt(Sdwcor)+epsilon)
                    cb = np.divide(Vdbcor, np.sqrt(Sdbcor)+epsilon)
                    self.weights[l-1] -= alpha*cw
                    self.biases[l-1] -= alpha*cb
                t += 1
                #hf.progress_bar(j, int(len(train_X)*USED_SHARE_OF_DATA/BATCH_SIZE), EPOCHS, i, round(100*self.correct/(self.used), 2))
            testAcc, testLoss = hf.test(self, 10000, test_X, test_y)
            trainAcc = self.correct/self.used
            trainAccVec.append(trainAcc) 
            testAccVec.append(testAcc)
            testLossVec.append(testLoss)
            
        #loss = hf.averageLoss(self.lossFuncValues)
        return trainAccVec, trainLossVec, testAccVec, testLossVec