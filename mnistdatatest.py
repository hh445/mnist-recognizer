from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np


mnist = fetch_openml('mnist_784',version =1,as_frame=False,parser='liac-arff')

data = np.transpose(mnist.data)

X = data[:,155]/255

image = X.reshape((28,28))

plt.imshow(image,cmap='gray',vmin=0,vmax=1)

plt.show()
#plt.plot(x,y,color = 'blue', marker = 'o', linestyle = '--')

