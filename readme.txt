readme.txt

1. Organization
This Matlab package includes four source files: pcn.m, mlp.m, knn.m and main.m, the description of each file is below:
	1) pcn.m: single layer neuro network perceptron, the inputs parameters are "Inputs, Targets, eta, nIteration", and outputs parameters are "Weights,Activation".
		Inputs: training data, each row for a datapoint
		Targets: training targets
		eta: learning rate
		nIteration: iteration number of optimization
	2) mlp.m:  multiple layer neuro network perceptron, the inputs parameters are "inputs, targets, testIn, nhidden, eta, nIteration, beta, momentum", and outputs parameters are "outTest".
		inputs: training data, each row for a datapoint
		targets: training targets
		testIn: testing data, each row for a datapoint
		nhidden: number of perceptrons in the hidden layer
		eta: learning rate
		beta: weight adjust rate
		momuntum: momumtum by practice is 0.8
		outTest: classified result of testing data
	3) knn.m: k-nearest-neighbour. The inputs parameters are "trainedData,dataClass,inputs,k", the output parameter is "closest".
		trainedData: training data, each row for a datapoint
		dataClass: training targets
		inputs: testing data, each row for a datapoint
		k: default as 1, or can be configured by inputs
		closest: classified result of testing data
	4) main.m: the main program including importing data, data preparation/ feature selection, inputs model comparasion, leave-one-out validation, figure plotting.

2. Code block
-In the main.m, the code is arranged in blocked and each block is commented with specific usage. User may refer to the comments for instructions.

3. Basic test
-Code block 1 is used to test basic pcn, mlp, knn algorithm.

4. Data import
-Code block 2 is used to import data. 
-For convenience, please put the data dir in the same directory with this code. And please rename the rawdata.csv to index_class.csv, the index starts from 1 and is consecutive, for example, 1_0, 2_0, 3_0, 4_0, 1_1, 2_1, 3_1... 
-User needs to adjust the parameter formatSpec0 and formatSpec1 with the corresponding data directory and maxNumData which is the max(numClass0, numClass1).

5. Feature selection
-Code block 3 is used to try different feature 
-This will be discussed in details in report

6. inputs model comparasion & leave-one-out validation
-Code block 4 is used for two things.
-The outter loop is used to try different combination of inputs, weight W changes so the inputs combination is different in this loop, and the accuracy is calculated to find out the best inputs combination. 
-The inner loop is used to apply leave-one-out validation. The code is designed as k-cross-fold validation and I set kfold=n which is 39 here so it can be treated as leave-one-out validation. User can easily modify kfold to other number to try k-cross-fold validation

7. ROC plotting & threshold selection
-Code block 5 is used to plot ROC and select best threshold for mlp
-Choose best inputs combination based on result of Code block 4, set p and x accordingly
-Since different run may yield different results, the outter layer is used to get specific data, the inner layer is to compute related staistic parameters 






















