import math
import numpy as np
import sys
import statistics
# Helpfuncs contain functions to assist the neural network
class Helpfuncs:
    # Tests the network with test images
    def test(NN, test_X, test_y):
        correct = 0
        for i in range(0,len(test_X)):
            predictionVec = NN.predict(test_X[i])
            prediction = max(predictionVec)
            index = predictionVec.index(prediction)
            if (index == int(test_y[i])):
                correct += 1
        acc =  round(correct/len(test_X),4)
        print('Testing accuracy: ', acc)
        return acc
    # Sorts matrix into vector
    def flatten(matrix):
        rows = len(matrix)
        columns = len(matrix[0])
        array = [0]*rows*columns
        for row in range(0,rows):
            array[row*columns:(rows+1)*columns-1]= matrix[row][:]
        return array
    # Transforms the value of Y into a vector with 0 for incorrect answers and 1 for the correct answer
    def formatY(number):
        yvec = []
        for i in range(0,10):
            if (i == number):
                yvec.append(1.0)
            else:
                yvec.append(0.0)
        yvec = np.array(yvec)
        return yvec
    # Sigmoid activation function for a vector x
    def sigmoid(x):
        a = []
        for i in x:
            f = 1/(1+np.exp(-i))
            a.append(f)
        return a 
    # Derivative of sigmoid activation function for a vector x
    def sigmoidPrim(x):
        x = Helpfuncs.sigmoid(x)
        a = []
        for i in x:
            a.append(i*(1-i))
        return a
    # Relu activation function max(0,x) for a vector x
    def relu(x):
        a = []
        for i in x:
            if i > 0:
                a.append(i)
            else:
                a.append(0)
        return a
    # Derivative of relu activation function for a vector x
    def reluPrim(x):
        aprim =[]
        for i in x:
            if i > 0:
                aprim.append(1)
            else:
                aprim.append(0.01)
        return aprim
    # Calculates the loss
    def lossFunc(x, y):
        yvec = Helpfuncs.formatY(y)
        return sum(np.square(x-yvec))
    # exponentialDecay stepping schedule update
    def exponentialDecay(initialLR, decayRate, epoch):
        return initialLR*decayRate ** (epoch)
    # piecewiseConstantDecay stepping schedule update
    def piecewiseConstantDecay(initialLR, epoch):
        LRvec = initialLR*np.array((1, 0.909, 0.818, 0.727, 0.636, 0.545, 0.4545, 0.3636, 0.2727, 0.1818, 0.0909))
        return LRvec[epoch]
    # polynomialDecay stepping schedule update
    def polynomialDecay(initialLR, endLR, epoch, decaySteps):
        power = 0.5
        return (initialLR - endLR) * (1 - epoch/decaySteps) ** (power) + endLR
    # inverseTimeDecay stepping schedule update
    def inverseTimeDecay(initialLR, decayRate, epoch):
        return initialLR/(1 + decayRate*epoch)
    # Calculates the averages of loss function sections
    def averageLoss(allLosses, numOfAverages = 50):
        l = len(allLosses)
        numInAverages = math.ceil(l/numOfAverages)
        loss = []
        for i in range(0,numOfAverages-1):
            try:
                loss.append(statistics.mean(allLosses[i*numInAverages:(i+1)*numInAverages]))
            except:
                if i*numInAverages < l:
                    loss.append(statistics.mean(allLosses[i*numInAverages:]))
                else:
                    break   
        return loss
    # Prints a progress bar to the terminal
    def progress_bar(curr_ite, total, total_epoch, curr_epoch, acc, prefix = '', decimals = 0, barLength = 20):
        '''
        Prints the progress of a for loop.
        I have used the modified version of Greenstick code on stackoverflow.
        Link to the code: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?answertab=votes#tab-top
        :param curr_ite: current iteration
        :param total: total iterations
        :param total_epoch: total epochs to be conducted
        :param curr_epoch: current epoch
        :param prefix: not required(default Progress)
        :param decimals: positive number of decimals in percent complete (Int)
        :param barLength: length of the bar
        :return: None
        '''
        formatStr = "{0:." + str(decimals) + "f}"
        percent = formatStr.format(100 * (curr_ite / float(total)))
        filledLength = int(round(barLength * curr_ite / float(total)))
        bar = '=' * filledLength + '-' * (barLength - filledLength)
        sys.stdout.write('\rEpoch %s/%s %s %s %s%s %s %s%s' % (curr_epoch,total_epoch,prefix, bar, percent, '%', "Training accuracy", acc, '%')),
        if curr_ite == total:
            sys.stdout.write('\n')
        sys.stdout.flush()