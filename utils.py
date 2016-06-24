#!/usr/bin/python2

"""
N Layer Network
```````````````
This is a class for instantiating N layered networks. you can specify various 
hyper parameters at initialisation like training cost functions, training 
loops, alpha values etc...

Very largely inspired by: mnielsen
https://github.com/mnielsen/neural-networks-and-deep-learning/
blob/master/src/network2.py

I have made modifications to allow different activation functions for 
different layers and will hopefully be making this an even more useful set
of classes/methods for exploring machine learning via ANN's.
"""

import numpy as np
import json
import random
import sys

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
return e