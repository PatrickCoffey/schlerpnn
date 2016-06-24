#!/usr/bin/python2

"""
Activation functions
````````````````````
These are my activation functs, these basically control whether or not a 
neuron will fire.

eg, pass it an x and it will output a y

Very largely inspired by: mnielsen
https://github.com/mnielsen/neural-networks-and-deep-learning/
blob/master/src/network2.py
"""


import numpy as np

class sigmoid(object):
    
    def fx(self, x):
        try:
            return 1/(1+np.exp(-x))
        except FloatingPointError:
            return np.zeros(x.shape)
    
    def ffx(self, x):
        try:
            return x*(1-x)
        except FloatingPointError:
            return np.zeros(x.shape)

    #def fffx(self, x):
        #pass


class tanh(object):
    
    def __init__(self, y_shift=0.0):
        self.y_shift = y_shift
    
    def fx(self, x):
        try:
            return np.tanh(x) + self.y_shift
        except FloatingPointError:
            return 0.0
    
    def ffx(self, x):
        try:
            return 1.0-x**2 + self.y_shift
        except FloatingPointError:
            return 0.0

    #def fffx(self, x):
        #pass


class relu(object):
    
    def fx(self, x):
        try:
            return x*(abs(x))
        except FloatingPointError:
            return 0.0
    
    def ffx(self, x):
        try:
            return 2*(abs(x))
        except FloatingPointError:
            return 0.0

    #def fffx(self, x):
        #pass


class softmax(object):
    
    def fx(self, x):
        try:
            return np.exp(x) / np.sum(np.exp(x))
        except FloatingPointError:
            return 0.0
    
    def ffx(self, x):
        try:
            return 2*(np.exp(x) / np.sum(np.exp(x)))
        except FloatingPointError:
            return 0.0

    #def fffx(self, x):
        #pass
        
if __name__ == '__main__':
    """
    lets do some testing....
    activatiopn funcs seem to work! :)
    """
    #x = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], dtype=np.float)
    #print(x)
    #act = act_sigmoid()
    #fx = act.fx(x)
    #ffx = act.ffx(x)
    #print(fx)
    #print(ffx)
    
    #import matplotlib.pyplot as plt
    #plt.plot(x)
    #plt.title('values for x')
    #plt.ylabel('x')
    #plt.show()
    
    #plt.plot(fx)
    #plt.ylabel('fx')
    #plt.title('values for f(x)')
    #plt.show()    

    #plt.plot(ffx)
    #plt.ylabel('ffx')
    #plt.title('values for ff(x)')
    #plt.show()