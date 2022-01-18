import numpy
import scipy.special
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
#%matplotlib inline

# Strategy: Fix the names of the variables. DONE!
# Is the code meaningful? 
# Why do we need so many comments? Clarify code by chosing more verbose names weightInToHidden etc
# Consider porting to another language.
# Read up docs on specific Python libraries.




class NeuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
 
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        
        # weight: input to hidden
        self.wih = numpy.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes, self.iNodes))
        
        # weight: hidden to output
        self.who = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.oNodes, self.hNodes))
        
        self.lr = learningRate
        
        # activation function is the sigmoid function
        self.activationFunction = lambda x: scipy.special.expit(x)
        
        pass

    

    def train(self, inputsList, targetsList):
        
        # convert inputs list to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T
        
        # calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        
        # calculate the signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        # calculate signals into final output layer
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # calculate the signals emerging from final output layer
        finalOutputs = self.activationFunction(finalInputs)
        
        # output layer error is the (target - actual)
        outputErrors = targets - finalOutputs
        
        # hidden layer error is the outputErrors, split by weights, recombined at hidden nodes
        hiddenErrors = numpy.dot(self.who.T, outputErrors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))
        
        pass

    

    def query(self, inputsList):
        # convert inputs list to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T
        
        # calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        # calculate signals into final output layer
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # calculate the signals emerging from final output layer
        finalOutputs = self.activationFunction(finalInputs)
        
        return finalOutputs
   
#################################################
         
        
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

    
        
#################################################
        

inputNodes = 784
hiddenNodes = 200
outputNodes = 10
learningRate = 0.1

debug = False
debug = True


#if debug:

#        trainingDataFileName = "mnist_dataset/mnist_train_100.csv"
        
#        testDataFileName = "mnist_dataset/mnist_test_10.csv"
#else:        
        
trainingDataFileName = "mnist_dataset/mnist_train.csv"

testDataFileName = "mnist_dataset/mnist_test.csv"




n = NeuralNetwork(inputNodes,hiddenNodes,outputNodes, learningRate)

trainingDataFile = open(trainingDataFileName, 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()


# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    
    print("Training epoch = ",e)
 
    for record in trainingDataList:

        allValues = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(outputNodes) + 0.01
        # allValues[0] is the target label for this record
        targets[int(allValues[0])] = 0.99
        n.train(inputs, targets)
        pass
        
    pass


testDataFile = open(testDataFileName, 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

scoreCard = []


myCounter = 0


for record in testDataList:

    allValues = record.split(',')

    correctLabel = int(allValues[0])


    inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)
    
    

   
    
    myCounter = myCounter + 1

    if (label == correctLabel):

        scoreCard.append(1)

    else:

        scoreCard.append(0)
    
        print(myCounter,": correctLabel = ",correctLabel," label = ",label)

        pass
    
    pass



scoreCard_array = numpy.asarray(scoreCard)
print ("learningRate = " , learningRate, " performance = ", scoreCard_array.sum() / scoreCard_array.size)



        
        
        