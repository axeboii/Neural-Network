%  This MATLAB problem visualises data produced with associated python
%  program, called "TestingMain.py". It is divided into 8 parts to 
%  visualise different aspects of the neural network. Please see 
%  comments in code for details.
% 
%  A folder "results" with datais required, as well as MATLAB function
%  file "smoothen.m"
% 
%  Authors: Axel Boivie <aboivie@kth.se>
%           Victor Bellander <vicbel@kth.se>
%          
%  Course:  SA115X Degree Project in Vehicle Engineering, 
%           First Level, 15.0 Credits, KTH
%
%  Project: Implementation and Optimisation of a Neural Network
%           Spring 2021, KTH, Stockholm
%
%  GitHub:  https://github.com/axeboii/NeuralNetwork_SA115X
%  Last modified 2021/04/19 by AB

%% 1. Plot with different learning rates
close all;clear;
folder = "results/";
name = "noScheme";
lrs = [0.05, 0.3, 0.75];
dr = 0;
% Loop through learning rate
for lr = lrs
    % Load data
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;    
    % Plot
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on
end
figure(1)
title("Training loss ")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss ")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy ")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy ")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Accuracy")

%% 2. Comparing stepping schedules
close all;clear;
lrs = [0.3, 0.5, 0.5, 0.3, 0.3];
drs = [0, 0.75, 0.5, 0, 2.5];
schemes = ["noScheme", "exponentialDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay"];
% lrs = [0.3, 0.5, 0.003];
% drs = [0, 0.5, 0];
% schemes = ["noScheme", "inverseTimeDecay", "ADAM"];
folder = "results/";
size = ""; %["", "_1616", "_400400"]
% Loops through the schemes
for i = 1:length(schemes)
    name = schemes(i);
    lr = lrs(i);
    dr = drs(i);
    % Load data
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+size+".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;    
    % Plot
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on
end
figure(1)
title("Training loss")
legend(schemes)
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss")
legend(schemes)
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy")
legend(schemes)
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy")
legend(schemes)
xlabel("Batch")
ylabel("Accuracy")
% Plot learning rates
figure(5)
epoch = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
noScheme = 0.3*ones(20,1);
ed = 0.5*(0.75.^epoch);
itd = 0.5./(1+0.5*epoch);
pcd = 0.3*[1, 1, 0.909, 0.909, 0.818, 0.818, 0.727, 0.727, 0.636, 0.636,0.545, 0.545, 0.4545, 0.4545, 0.3636, 0.3636, 0.2727, 0.2727, 0.1818, 0.1818];
pd = (0.3 - 0.01)*(1 - epoch/10).^2.5 + 0.01;
b = 375;
x = [0, b, b+1, 2*b, 2*b+1, 3*b, 3*b+1, 4*b, 4*b+1, 5*b, 5*b+1, 6*b, 6*b+1, 7*b, 7*b+1, 8*b, 8*b+1, 9*b, 9*b+1, 10*b];
plot(x, noScheme, x, ed, x, itd, x, pcd, x, pd)
legend(schemes)
xlabel("Batch")
ylabel("Learning rate")
title("Learning rates")

%% 3. Compare network sizes
close all; clear;
lrs = [0.3, 0.5, 0.5, 0.3, 0.3, 0.003];
drs = [0, 0.75, 0.5, 0, 2.5, 0];
schemes = ["noScheme", "exponentialDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay", "ADAM"];
% Pick stepping scheme
schemeVariant = 3;
lr = lrs(schemeVariant);
dr = drs(schemeVariant);
name = schemes(schemeVariant);
sizes = ["", "_1616", "_400400"];
folder = "results/";
% Loops through network sizes
for size = sizes
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr) + size +".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;    
    % Plot
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on    
end
figure(1)
title("Training loss - " + name)
legend("32", "16-16", "400-400")
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss - " + name)
legend("32", "16-16", "400-400")
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy - " + name)
legend("32", "16-16", "400-400")
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy - " + name)
legend("32", "16-16", "400-400")
xlabel("Batch")
ylabel("Accuracy")

%% 4. Compare all schemes with ADAM
close all;clear;
lrs = [0.3, 0.5, 0.5, 0.3, 0.3, 0.003];
drs = [0, 0.75, 0.5, 0, 2.5, 0];
schemes = ["noScheme", "exponentialDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay", "ADAM"];
folder = "results/";
size = ""; % Or use ["", "_1616", "_400400"]
% Loop through stepping schedules
for i = 1:length(schemes)
    name = schemes(i);
    lr = lrs(i);
    dr = drs(i);
    % Load data
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+size+".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;
    % Plot
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on
end
figure(1)
title("Training loss" + size)
legend(schemes)
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss" + size)
legend(schemes)
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy" + size)
legend(schemes)
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy" + size)
legend(schemes)
xlabel("Batch")
ylabel("Accuracy")

%% 5. Plot ITD (constant decay rate)
close all;clear;
folder = "results/";
name = "inverseTimeDecay";
lrs = [0.05, 0.5, 0.75];
drs = [0.1, 0.5, 2];
dr = drs(2);
% Loop through learning rates
for lr = lrs
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on
end
figure(1)
title("Training loss \gamma = 0.5")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss \gamma = 0.5")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy \gamma = 0.5")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy \gamma = 0.5")
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Accuracy")
% Plot learning rates
figure(5)
epoch = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
itd1 = 0.05./(1+0.5*epoch);
itd2 = 0.5./(1+0.5*epoch);
itd3 = 0.75./(1+0.5*epoch);
b = 375;
x = [0, b, b+1, 2*b, 2*b+1, 3*b, 3*b+1, 4*b, 4*b+1, 5*b, 5*b+1, 6*b, 6*b+1, 7*b, 7*b+1, 8*b, 8*b+1, 9*b, 9*b+1, 10*b];
plot(x, itd1, x, itd2, x, itd3)
legend("\epsilon_0 = " + num2str(lrs(1)), "\epsilon_0 = " + num2str(lrs(2)), "\epsilon_0 = " + num2str(lrs(3)))
xlabel("Batch")
ylabel("Learning rate")
title("Learning rate \gamma = 0.5")

%% 6. Plot ITD (constant learning rate)
close all;clear;
folder = "results/";
name = "inverseTimeDecay";
lrs = [0.05, 0.5, 0.75];
drs = [0.1, 0.5, 2];
lr = lrs(2);
% Loop through decay rate
for dr = drs
    fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+".dat";
    trainLoss = load(folder + "TrainLoss_" + fileEnd);
    testLoss = load(folder + "TestLoss_" + fileEnd);
    trainAcc = load(folder + "TrainAcc_" + fileEnd);
    testAcc = load(folder + "TestAcc_" + fileEnd);
    % Averages the 10 runs
    trainLoss = mean(trainLoss,2);
    testLoss = mean(testLoss,2);
    trainAcc = mean(trainAcc,2);
    testAcc = mean(testAcc,2);
    % Condenses to less datapoints
    trainLoss = smoothen(trainLoss, 75, 50);
    testLoss = smoothen(testLoss, 75, 50);
    trainAcc = smoothen(trainAcc, 75, 50);
    testAcc = smoothen(testAcc, 75, 50);
    % Fix x-axis to batches
    step = 3750/(length(trainLoss)-1);
    x = 0:step:3750;
    % Plot
    figure(1)
    plot(x, trainLoss)
    hold on
    figure(2)
    plot(x, testLoss)
    hold on
    figure(3)
    plot(x, trainAcc)
    hold on
    figure(4)
    plot(x, testAcc)
    hold on
end
figure(1)
title("Training loss \epsilon_0 = 0.5")
legend("\gamma = " + num2str(drs(1)), "\gamma = " + num2str(drs(2)), "\gamma = " + num2str(drs(3)))
xlabel("Batch")
ylabel("Loss")
figure(2)
title("Testing loss \epsilon_0 = 0.5")
legend("\gamma = " + num2str(drs(1)), "\gamma = " + num2str(drs(2)), "\gamma = " + num2str(drs(3)))
xlabel("Batch")
ylabel("Loss")
figure(3)
title("Training accuracy \epsilon_0 = 0.5")
legend("\gamma = " + num2str(drs(1)), "\gamma = " + num2str(drs(2)), "\gamma = " + num2str(drs(3)))
xlabel("Batch")
ylabel("Accuracy")
figure(4)
title("Testing accuracy \epsilon_0 = 0.5")
legend("\gamma = " + num2str(drs(1)), "\gamma = " + num2str(drs(2)), "\gamma = " + num2str(drs(3)))
xlabel("Batch")
ylabel("Accuracy")
% Plot learning rates
figure(5)
epoch = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
itd1 = 0.5./(1+0.1*epoch);
itd2 = 0.5./(1+0.5*epoch);
itd3 = 0.5./(1+2*epoch);
b = 375;
x = [0, b, b+1, 2*b, 2*b+1, 3*b, 3*b+1, 4*b, 4*b+1, 5*b, 5*b+1, 6*b, 6*b+1, 7*b, 7*b+1, 8*b, 8*b+1, 9*b, 9*b+1, 10*b];
plot(x, itd1, x, itd2, x, itd3)
legend("\gamma = 0.1", "\gamma = 0.5", "\gamma = 2")
xlabel("Batch")
ylabel("Learning rate")
title("Learning rate \epsilon_0 = 0.5")

%% 7. Testing vs training loss/accuracy
close all;clear;
% Learning rate/decay rate of all schemes
lrs = [0.3, 0.5, 0.5, 0.3, 0.3, 0.003];
drs = [0, 0.75, 0.5, 0, 2.5, 0];
schemes = ["noScheme", "exponentialDecay", "inverseTimeDecay", "piecewiseConstantDecay", "polynomialDecay", "ADAM"];
sizes = ["", "_1616", "_400400"];
% Pick the scheme
schemeVariant = 1;
lr = lrs(schemeVariant);
dr = drs(schemeVariant);
name = schemes(schemeVariant);
% Pick the network size
sizeVariant = 1;
size = sizes(sizeVariant);
folder = "results/";
% Load data
fileEnd = name + "_" + num2str(lr) + "_" + num2str(dr)+size+".dat";
trainLoss = load(folder + "TrainLoss_" + fileEnd);
testLoss = load(folder + "TestLoss_" + fileEnd);
trainAcc = load(folder + "TrainAcc_" + fileEnd);
testAcc = load(folder + "TestAcc_" + fileEnd);
% Averages the 10 runs
trainLoss = mean(trainLoss,2);
testLoss = mean(testLoss,2);
trainAcc = mean(trainAcc,2);
testAcc = mean(testAcc,2);
% Condenses to less datapoints
trainLoss = smoothen(trainLoss, 75, 50);
testLoss = smoothen(testLoss, 75, 50);
trainAcc = smoothen(trainAcc, 75, 50);
testAcc = smoothen(testAcc, 75, 50);
% Fix x-axis to batches
step = 3750/(length(trainLoss)-1);
x = 0:step:3750;    
% Plot
figure(1)
plot(x, trainLoss)
hold on
plot(x, testLoss)
xlabel("Step")
ylabel("Loss")
legend("Training", "Testing")
title("Losses")
figure(2)
plot(x, trainAcc)
hold on
plot(x, testAcc)
xlabel("Step")
ylabel("Accuracy")
legend("Training", "Testing")
title("Accuracies")

%% 8. Plot large run 
close all; clear;
lr = 0.3;
dr = 0;
scheme = "noScheme";
size = "_400400_big";
% Load data
folder = "results/";
fileEnd = scheme + "_" + num2str(lr) + "_" + num2str(dr) + size +".dat";
trainLoss = load(folder + "TrainLoss_" + fileEnd);
testLoss = load(folder + "TestLoss_" + fileEnd);
trainAcc = load(folder + "TrainAcc_" + fileEnd);
testAcc = load(folder + "TestAcc_" + fileEnd);
% Condenses to less datapoints
trainLoss = smoothen(trainLoss, 375, 500);
testLoss = smoothen(testLoss, 375, 500);
trainAcc = smoothen(trainAcc, 375, 500);
testAcc = smoothen(testAcc, 375, 500);
% Fix x-axis to batches
step = 3750/(length(trainLoss)-1);
x = 0:step:3750;
% Plot
figure(1)
plot(x, (trainLoss))
hold on
plot(x, (testLoss))
xlabel("Step")
ylabel("Loss")
title("Loss")
legend("Training", "Testing")
figure(2)
plot(x, trainAcc)
hold on
plot(x, testAcc)
xlabel("Step")
ylabel("Accuracy")
title("Accuracy")
legend("Training", "Testing")
