import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from pathlib import Path

mnist = fetch_openml('mnist_784',version =1,as_frame=False,parser='liac-arff') #importing mnist datasett
data = np.transpose(mnist.data)/255.0
ref = np.transpose(np.eye(10)[mnist.target.astype(int)])
rng = np.random.default_rng()

script_dir = Path(__file__).resolve().parent
saved_parameters = np.load(script_dir / "mnist_recog_params.npz")

layers = [784,128,64,10]

def softmax(z):
    #its kinda st andard to do z - np.max to decrease all the exponents by the maximum number. It helps keeps number bounded without them blowing up because of the huge exponents and does not change the outcome.
    ex = np.exp(z - np.max(z,axis = 0,keepdims= True))
    return ex/np.sum(ex)

weights = [saved_parameters[f'W{i}'] for i in range(len(layers)-1)] #weight matrix for every layer, where the columns correspond to the amoutn of input vectors and rows correspond to amount of amount of output vectors (it maps high dimensional input vectors into a lower dimension output vector in my case.)
#biases are started off with 0 because the weights are already random. however, the weights cannot be 0 because if they are all of the outputs of the first layer would be zero because the input vectors are just multiplying by a zero matrix, so zero output.
biases = [saved_parameters[f'b{i}'] for i in range(len(layers)-1)]

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

ind = rng.integers(0,60001) #random number between 0 and 60000 to pull from the mnist dataset
out, active = forwardFeed(data[:,ind:ind+1]) #feed into the neural net

print(np.round(out,2))
max = 0
highIndex = 0
for index, logit in enumerate(out):
    if logit > highIndex:
        highIndex = logit
        max = index
print("neural net says its " + str(max))
image = data[:,ind:ind+1].reshape((28,28))
plt.imshow(image,cmap='gray',vmin=0,vmax=1)

plt.show()