import numpy as np
from sklearn.datasets import fetch_openml

#i program sparsely between like 3 languages so i dont really know naming conventions for each specific lang and i dont program often so my code might be messy or unorganiazed please ignore
mnist = fetch_openml('mnist_784',version =1,as_frame=False,parser='liac-arff') #importing mnist datasett
data = np.transpose(mnist.data)/255.0
ref = np.transpose(np.eye(10)[mnist.target.astype(int)])

rng = np.random.default_rng()


layers = [784,128,64,10]
LEARNING_RATE = 0.01
BATCH_SIZE = 100
TRAINING_EPOCHS = 15
DATA_SIZE = 60000 #TRY TO MAKE IT SO THAT DATA_SIZE IS DIVISIBLE BY BATCH SIZE


#since this neural net is using relu, so the outputs are numbers which arent contained by any activation function. softmax compresses it into probabilities that sum up to 1, depending on their difference. More positive closer to 1, basically more likely to be that choice
def softmax(z):
    #its kinda st andard to do z - np.max to decrease all the exponents by the maximum number. It helps keeps number bounded without them blowing up because of the huge exponents and does not change the outcome.
    ex = np.exp(z - np.max(z,axis = 0,keepdims= True))
    return ex/np.sum(ex)

#
def softmax_cel():
    pass
#im using kaiming initialization, since this neural network uses relu half of signals are dropped.

#intializing the weights with more variance prevents vanishing gradients (where some weights are never trained)
#x is number of neurons in the previous layer which kaiming intialization uses
#the random generation is a normal distribution around 0, with a variance of the kaiming thingy
weights = [(np.sqrt(2/x))*(np.random.randn(y,x)) for x,y in zip(layers[:-1],layers[1:])] #weight matrix for every layer, where the columns correspond to the amoutn of input vectors and rows correspond to amount of amount of output vectors (it maps high dimensional input vectors into a lower dimension output vector in my case.)
#biases are started off with 0 because the weights are already random. however, the weights cannot be 0 because if they are all of the outputs of the first layer would be zero because the input vectors are just multiplying by a zero matrix, so zero output.
biases = [np.zeros((x,1)) for x in layers[1:]]

#this function is the kind of the core of the neural net it is the function that takes in teh input and passes it through the network generating an output
#every layer the input vector is multiplied by the weight matrix, giving an output vector that is then added by the bias.
def forwardFeed(input):
    #the for loop iterate through all the layers, but it runs 1 less than the amoutn of layers because only n-1 calculations are needed, (the amoutn of gaps between layers)
    activations = []
    for i in range(len(layers)-1):
        if i < len(layers)-2:
            activations.append(input) #the activations 
            input = np.maximum(0,(np.dot(weights[i],input) + biases[i]))
        else:
            activations.append(input)
            input = softmax(np.dot(weights[i],input) + biases[i])

    return input, activations

#sm is the softmax output from the neural network, while ref is the one hot coded reference to the correct answer
def backprop(sm,ref,activ):
    backcarry = sm - ref #the rates of change in respect oteh loss of the olutput logits. This will be used to propogate backwards
    weightGrad = []
    biasGrad = []
    for x,y in zip(reversed(weights),reversed(activ)):
        #here is where i pull the gradients (yet to be implemented)
        weightGrad.append(np.dot(backcarry,y.reshape(1,-1))) #the gradients is the product of the matrix multiplication between the column vector of the back carry and the row vector of the previous layer activations. I will explain elsewhere too little space in this comment.
        biasGrad.append(backcarry) #the gradient of the bias is just the backcarry because derivative of a variable by itself is one.
        backcarry = np.dot(np.transpose(x),backcarry)*(y > 0)  #pass back the gradients of the neurons in respect othe loss function. This is the part of "back propogation" Again i will explain this in more depth elsewhere
    return weightGrad[::-1], biasGrad[::-1]

def gradientDescent(data,rate,batch_size): #the parameter data is zipped in a tuple of (x,y) where x is the actual image and y is the correct answer #rate is the learning rate (the multiplier at whic hteh gradient will be multiplied by)
    gradWeight= [np.zeros((y,x)) for x,y in zip(layers[:-1],layers[1:])]
    gradBias = [np.zeros((x,1)) for x in layers[1:]]
    count = 0
    for x,y in data:
        sftmx,activations = forwardFeed(x)
        is_max = (sftmx == sftmx.max())
        out = sftmx.copy()
        out[is_max] = 1
        out[~is_max]=0
        if np.array_equal(out,y):
            count += 1
        deltaW, deltaB = backprop(sftmx,y,activations)
        for i in range (len(weights)):
            gradWeight[i] += deltaW[i]/batch_size
            gradBias[i] += deltaB[i]/batch_size
    for i in range(len(weights)): #the actual part where the weights and biases are tweaked
        weights[i] -= rate*gradWeight[i] #gradient descent
        biases[i] -= rate*gradBias[i]  #gradient descent
    return count


#the actual training goes on down here

training_accuracy =[]
for j in range(TRAINING_EPOCHS): 
    startingNum = 0
    correctCount = 0
    shufflePerm = rng.permutation(DATA_SIZE) #makes it so every epoch goes through the data differently
    for k in range(round(DATA_SIZE/BATCH_SIZE)): #training the network
        training_data = []
        ref_data = []
        for i in range(BATCH_SIZE):
            index = shufflePerm[startingNum + i] #the shuffled index
            training_data.append(data[:,index:index+1])
            ref_data.append(ref[:,index : index+1])
        startingNum += BATCH_SIZE
        correctCount += gradientDescent(zip(training_data,ref_data),LEARNING_RATE,BATCH_SIZE) #where the gradient descent happens (all the fluff on top is just to sort data and input it in)
        if (k+1) % 100 == 0:
             print("step: " + str(k+1))
    training_accuracy.append(correctCount/DATA_SIZE)
    print("epoch: " + str(j+1))
    print("accuracy: " + str(correctCount/DATA_SIZE))

testCount = 0
for i in range(10000): #mnist has 70k in total, so the last 10k is used as testing data. training data uses the first 60k (atlaest thats how they do it usually)
    out, active = forwardFeed(data[:, i + DATA_SIZE: i + DATA_SIZE + 1])
    is_max = (out == out.max())
    out[is_max] = 1
    out[~is_max] = 0
    if np.array_equal(out, ref[:, i + DATA_SIZE: i+ DATA_SIZE + 1]):
        testCount+= 1
print("TEST ACCURACY: " + str(testCount/10000))
np.savez('mnist_recog_params.npz',
         **{f'W{i}': w for i, w in enumerate(weights)},
         **{f'b{i}': b for i, b in enumerate(biases)}
         )

    







