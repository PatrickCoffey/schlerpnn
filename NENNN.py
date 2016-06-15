#!/usr/bin/python

# Patty's neural network
# ----------------------
#
# a bunch of shit for me to fuck around with and learn 
# neural networks.
# currently idea is:
#  * pick a non linear
#  
#  * initialise a network with the amount of input,
#    hidden, and output nodes that you want for the 
#    data
#  
#  * train the network with training x and y
#  
#  * test it using nn.guess(x) with a known y
#  
#  * if happy, test with real world data :P
#


# example 6 layer nn for solving mnist
# i  -h   -h   -h   -h   -h  -o
# 784-2500-2000-1500-1000-500-10

import numpy as np
import numexpr as ne
import struct
import os


# =============================================================
# classes

class NENonlinear(object):
    """
    Nonlinear
    ---------
    this is used to set up a non linear for a network. The idea is you can
    instantiate it and set what type of non linear function it will be for
    that particular neaural network
    """
    
    _FUNC_TYPES = ('sigmoid',
                   'softmax',
                   'relu',
                   'tanh',
                   'softplus')
    
    def __init__(self, func_type='sigmoid'):
        if func_type in self._FUNC_TYPES:
            if func_type == self._FUNC_TYPES[0]:
                # sigmoid
                self._FUNCTION = self._FUNC_TYPES[0]
            elif func_type == self._FUNC_TYPES[1]:
                # softmax
                self._FUNCTION = self._FUNC_TYPES[1]
            elif func_type == self._FUNC_TYPES[2]:
                # relu
                self._FUNCTION = self._FUNC_TYPES[2]
            elif func_type == self._FUNC_TYPES[3]:
                # tanh
                self._FUNCTION = self._FUNC_TYPES[3]
            elif func_type == self._FUNC_TYPES[4]:
                # softplus
                self._FUNCTION = self._FUNC_TYPES[4]
        else:
            # default to sigmoid on invalid choice?
            print("incorrect option `{}`".format(func_type))
            print("defaulting to sigmoid")
            self._init_sigmoid()
    
    def __call__(self, x, derivative=False):
        ret = None
        if self._FUNCTION == self._FUNC_TYPES[0]:
            # sigmoid
            if derivative:
                ret = ne.evaluate('x*(1-x)')
            else:
                try:
                    ret = ne.evaluate('1/(1+exp(-x))')
                except RuntimeWarning:
                    ret = 0.0
        elif self._FUNCTION == self._FUNC_TYPES[1]:
            # softmax
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = ne.evaluate('2*(exp(x)/sum(exp(x)))')
            else:
                # from: https://gist.github.com/stober/1946926
                #e_x = np.exp(x - np.max(x))
                #ret = e_x / e_x.sum()
                # from: http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
                ret = ne.evaluate('exp(x)/sum(exp(x), axis=0)')
        elif self._FUNCTION == self._FUNC_TYPES[2]:
            # relu
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = ne.evaluate('2*(abs(x))')
            else:
                ret = ne.evaluate('x*(abs(x))')
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # tanh
            if derivative:
                # from my own memory of calculus :P
                ret = ne.evaluate('1.0-x**2')
            else:
                ret = ne.evaluate('tanh(x)')
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # softplus
            if derivative:
                # from wikipedia
                ret = ne.evaluate('1.0/(1+exp(-x))')
            else:
                ret = ne.evaluate('log(1+exp(x))')
        return ret


class NENNN(object):
    """NumExpr N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
        self.trained_loops_total = 0
        self.train_loops = []
        self.inputs = inputs
        self.outputs = outputs
        self._ALPHA = alpha
        self._num_of_weights = len(weights)
        self._LAYER_DEFS = {}
        self.WEIGHT_DATA = {}
        self.LAYER_FUNC = {}
        self.LAYERS = {}
        for i in range(self._num_of_weights):
            #(in, out, nonlin)
            self._LAYER_DEFS[i] = {'in': weights[i][0],
                                   'out': weights[i][1],
                                   'nonlin': weights[i][2]}
        print(self._LAYER_DEFS)
        self._init_layers()
    
    def _init_layers(self):
        for i in range(self._num_of_weights):
            _in = self._LAYER_DEFS[i]['in']
            _out = self._LAYER_DEFS[i]['out']
            _nonlin = self._LAYER_DEFS[i]['nonlin']
            self.WEIGHT_DATA[i] = np.random.randn(_in, _out)
            self.LAYER_FUNC[i] = _nonlin
    
    def reset(self):
        self._init_layers()
        
    def update_alpha(self, alpha=1):
        self._ALPHA = alpha
    
    def _do_layer(self, prev_layer, next_layer, nonlin):
        """Does the actual calcs between layers :)"""
        ret = nonlin(np.dot(prev_layer, next_layer))
        return ret

    def train(self, x, y, train_loops=100):
        for j in range(train_loops):
            # set up layers
            prev_layer = x
            prev_y = y
            next_weight = None
            l = 0
            self.LAYERS[l] = x
            for i in range(self._num_of_weights):
                l += 1
                next_weight = self.WEIGHT_DATA[i]
                nonlin = self.LAYER_FUNC[i]
                current_layer = self._do_layer(prev_layer, next_weight, nonlin)
                self.LAYERS[l] = current_layer
                prev_layer = current_layer
            last_layer = current_layer
            #print(last_layer)
            #
            #layer2_error = y - layer2
            #layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            #layer1_error = layer2_delta.dot(self.w_out.T)
            #layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            #self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            #self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)              
            
            # calculate errors
            output_error = ne.evaluate('y - last_layer')
            output_nonlin = self.LAYER_FUNC[self._num_of_weights - 1]
            output_delta = output_error * output_nonlin(last_layer, derivative=True)

            prev_delta = output_delta
            prev_layer = last_layer
            for i in reversed(range(self._num_of_weights)):
                weight = self.WEIGHT_DATA[i]
                current_weight_error = prev_delta.dot(weight.T)
                current_weight_nonlin = self.LAYER_FUNC[i]
                current_weight_delta = current_weight_error * current_weight_nonlin(self.LAYERS[i], derivative=True)
                # backpropagate error
                self.WEIGHT_DATA[i] += self._ALPHA * self.LAYERS[i].T.dot(prev_delta)
                prev_delta = current_weight_delta
                
            # increment the train counter, so i can see how many 
            # loops my pickled nets have trained
            self.trained_loops_total += 1
            
            # output important info
            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                #print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                #print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                for i in range(3):
                    #print("Guess: ")
                    #print(last_layer[i])
                    #print("output delta: ")
                    #print(np.round(output_delta, 2))
                    #print("Guess (rounded), Actual: ")
                    #guess = tuple(zip(np.round(last_layer[i], 1), y[i]))
                    #j = 0
                    #for item in guess:
                        #if item[1] == 1.0:
                            #print("**ACTUAL")
                        #if item != (0.0, 0.0):
                            #print("number: {}".format(j))
                            #print("  {}".format(item))
                            #j += 1
                    #print(str(guess))
                    print("===================")
                    print("Guess (rounded): ")
                    random = np.random.randint(0, len(x) - 2)
                    print(np.round(last_layer[random + i], 0))
                    print("Actual: ")
                    print(y[random + i])
                    
        self.train_loops.append({'loops': train_loops,
                                 'cases': len(y),
                                 'alpha': self._ALPHA})      

    def guess(self, x):
        prev_layer = x
        prev_y = y
        next_weight = None
        l = 0
        self.LAYERS[l] = x
        for i in range(self._num_of_weights):
            l += 1
            next_weight = self.WEIGHT_DATA[i]
            nonlin = self.LAYER_FUNC[i]
            current_layer = self._do_layer(prev_layer, next_weight, nonlin)
            self.LAYERS[l] = current_layer
            prev_layer = current_layer
        last_layer = current_layer
        return last_layer
    
    def print_stats(self):
        print("total trained loops: ")
        print(self.trained_loops_total)
        print("trained loops layout: ")
        print(self.train_loops)
        print("inputs: ")
        print(self.inputs)
        print("outputs: ")
        print(self.outputs)
        print("alpha: ")
        print(self._ALPHA)
        print("layers: ")
        print(self._LAYER_DEFS)


# =============================================================
# DEFS

def normalize(vals, ranges):
    """
    Normalize
    ---------
    Will take a an array of data and a tuple of tuples contains the mins and
    maxes for each val. This will transforma ll data to a score of 0 to 1"""
    
    def norm(val, v_min, v_max):
        return (val-v_min)/(v_max - v_min)
    if len(vals[0]) != len(ranges):
        print("Error, values and ranges dont match!!")
        return None
    d_struct = []
    for row in vals:
        temp_row = zip(row, ranges)
        d_struct.append(temp_row)
    ret = []
    for row in d_struct:
        temp_row = []
        for col in row: 
            #print(col) 
            temp = norm(col[0]*1.0, col[1][0]*1.0, col[1][1]*1.0)
            temp_row.append(temp)
        ret.append(temp_row)
    return ret

def make_mnist_np(l_items=None):
    if l_items == None:
        l_items = [100, 500, 1000, 2000, 5000, 10000, 20000]
    for items in l_items:
        print("grabbing {} data...".format(items))
        t_in, t_out = get_flat_mnist(items=items, normalize=True)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)
        with open('mnist/tx{}'.format(items), 'wb+') as f:
            pickle.dump(x, f)
        with open('mnist/ty{}'.format(items), 'wb+') as f:
            pickle.dump(y, f)


# from: https://gist.github.com/akesling/5358964
# Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
# which is GPL licensed.
def read(dataset="training", path="./data"):
    """
    Python function for importing the MNIST data set. It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise(ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def get_flat_mnist(dataset="training", path="./mnist", items=60000, normalize=False):
    images = tuple()
    labels = tuple()
    i = 0
    for image in read(dataset, path):
        images += (image[1],)
        labels += (image[0],)
        i += 1
        if i == items:
            break
    flat_images = tuple()
    for image in images:
        flat_image = np.ndarray.flatten(image)
        flat_images += (flat_image,)
    #del images

    out_labels = tuple()
    # [0,1,2,3,4,5,6,7,8,9]
    for item in labels:
        if item == 0:
            out_labels += ([1,0,0,0,0,0,0,0,0,0],)
        elif item == 1:
            out_labels += ([0,1,0,0,0,0,0,0,0,0],)
        elif item == 2:
            out_labels += ([0,0,1,0,0,0,0,0,0,0],)
        elif item == 3:
            out_labels += ([0,0,0,1,0,0,0,0,0,0],)
        elif item == 4:
            out_labels += ([0,0,0,0,1,0,0,0,0,0],)
        elif item == 5:
            out_labels += ([0,0,0,0,0,1,0,0,0,0],)
        elif item == 6:
            out_labels += ([0,0,0,0,0,0,1,0,0,0],)
        elif item == 7:
            out_labels += ([0,0,0,0,0,0,0,1,0,0],)
        elif item == 8:
            out_labels += ([0,0,0,0,0,0,0,0,1,0],)
        elif item == 9:
            out_labels += ([0,0,0,0,0,0,0,0,0,1],)
    return flat_images, out_labels

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def reshape(image_array):
    i = 0
    temp_image = []
    temp_row = []
    for item in image_array:
        temp_row.append(item)
        if i == 27:
            temp_image.append(temp_row)
            temp_row = []
            i = 0
        else:
            i += 1
    return np.array(temp_image, dtype=np.int32)


if __name__ == '__main__':
    
    #-------------------------------------------
    # MNIST attempt 2
    # ```````````````
    
    # TURNING INTO STANDALONE TOOL!
    
    # neural network with 784 inputs, for flattened input of
    # mnist data.
    
    # need to do:
    #  * add a more menu like system to make this into a machine
    #    learning aid
    
    try:
        import cPickle as pickle
    except:
        import pickle
    
    # get data
    if input("load mnist training data? (y/N): ").lower() == 'y':
        load_d = input("  enter filename (eg. 500 = tx-500, ty-500): ")
        with open("mnist/tx{}".format(load_d), 'rb') as f:
            x = pickle.load(f)
        with open("mnist/ty{}".format(load_d), 'rb') as f:
            y = pickle.load(f)
    else:
        print("grabbing data...")
        t_in, t_out = get_flat_mnist(items=1000, normalize=False)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)        
    
    
    # load/init network
    load = input("load network? (y/N): ")
    net_loaded = False
    if load.lower() == 'y':
        fname = input("network filename: ")
        with open('nets/{}'.format(fname), 'rb') as f:
            nnn = pickle.load(f)
            net_loaded = True
    else:
        # set hypervariables
        i_input = 784 # this is how many pixel per image (they are flat)
        i_out = 10
        #even shrink
        weights = ((784, 512, NENonlinear('sigmoid')), 
                   (512, 256, NENonlinear('sigmoid')),
                   (256, 128, NENonlinear('sigmoid')),
                   (128, 64, NENonlinear('sigmoid')),
                   (64, 32, NENonlinear('sigmoid')),
                   (32, 16, NENonlinear('sigmoid')),
                   (16, 10, NENonlinear('sigmoid')))
    
        # smaller
        #weights = ((784, 256, NENonlinear('sigmoid')), 
                   #(256, 64, NENonlinear('sigmoid')),
                   #(64, 10, NENonlinear('sigmoid')))
        
        # initialise network
        print("initialising network...")
        #nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
        nnn = NENNN(inputs=i_input, 
                    weights=weights, 
                    outputs=i_out,
                    alpha=0.1)
        print("  network initialised!")
    
    if net_loaded:
        stats = input("print stats about current net? (y/N): ")
        if stats.lower() == 'y':
            nnn.print_stats()

    # train network
    alpha = input("change alpha ({}): ".format(nnn._ALPHA))
    if alpha not in (0, "", None):
        nnn.update_alpha(np.float(alpha))

    loops = input("how man training loops? (100): ")
    if loops in (0, "", None):
        loops = 100
    else:
        loops = int(loops)
        
    batches = input("process in batches of? (0 = all at once): ")
    if batches in (0, "", None):
        batches = 0
    else:
        batches = int(batches)

    if batches == 0:
        print("training network for {} loops, all samples at once".format(loops))
        nnn.train(x, y, loops)
    else:
        print("training network for {} loops, in batches of {} samples".format(loops, batches))
        split = int(np.ceil(len(x) / batches))
        for i in range(split):
            _from = batches*i
            _to = batches*(i+1)
            _x = x[_from: _to]
            _y = y[_from: _to]
            nnn.train(_x, _y, train_loops=loops)
            print("  training using samples from {} to {} of {}".format(_from, _to, len(x)))
            
    
    #show_image = input("Show image (x[0])? (y/N): ")
    #if show_image.lower() == "y":
        ##print(reshape(x[0]))
        #show(reshape(x[0]))


    # save network
    save = input("save network? (y/N): ")
    if save.lower() == 'y':
        fname = input("save network as: ")
        with open('nets/{}'.format(fname), 'wb+') as f:
            pickle.dump(nnn, f)
