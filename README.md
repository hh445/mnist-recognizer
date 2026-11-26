# mnist-recognizer

i made this as a weekend project, and as of now i haev finished up the mnist digit recognizer, it is built from scratch using numpy because i want to apply myself. 
if anyone sees this my code is not a good example of what good programming looks like. I have no naming conventions and organization is bad.

# DEPENDENCIES:
scikit for the mnist database

pandas or something because its apparently needed to import the scikit database

numpy obviously

# the network itself

### the main network is mnistRecog.py

it loads the parameters from mnist_recog_params.npz, (a numpy file containing ndarrays of all the weights and biases) make sure to keep the
parameter file in the same directory / level as the mnistRecog.py because thats how i configured the loader. You can tweak it if you want but the way i have it
gives the least headaches. 

Current parameters are for a network with 784 inputs (28x28 because mnist is 28x28 pixels), 128 first hidden layer and 64 neurons second hidden layer. and 10 output neurons. I traiend it using 15 epochs,
and testing accuracy is about 95.2 %. To be fair the digits that the network gets wrong are not by any means what would be considered legible handwriting. 
#### the network usese ReLU (Rectified Linear Unit) for its activation function, which goes into softmax and uses cross categorical entropy loss. The namings are really fancy but they are pretty simple once you break it down

oh yea and running mnistRecog.py pulls a random handwritten digit from the mnist database and plots it using matlab plot. Then the terminal outputs what number the network thinks it is.

### neuralnetOne.py is where you can train the network

neuralnetOne.py has all the code for training the network itself using gradient descent. you can edit all the parameters on the top, such as the batch size or how many training epochs or whatever.
to train a network, running neuralnetOne.py will initialize random weights using Kaiming He initialization (some chinese ai engineer), and run through the first 60k handwritten digits however many epochs it is configured to train to.
Then the program will save all the parameters into the file mnist_recog_params.npz. If you want ot keep multiple files once you run neuralnetOne.py once rename the file before running it again because if the namings are the same the saved
np files will overwrite eachother.

## its pretty cool. Deterministic algorithms can decipher the randomness of human input. cool ai math

# Resources:
https://www.youtube.com/watch?v=aircAruvnKk
https://www.youtube.com/watch?v=IHZwWFHWa-w
https://visionbook.mit.edu/backpropagation.html


#this is NOT MADE BY AI 
no point in me doing this if i used ai



