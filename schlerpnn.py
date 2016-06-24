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
import os


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



class Nonlinear(object):
    """
    Nonlinear
    ---------
    this is used to set up a non linear for a
    network. The idea is you can instantiate it 
    and set what type of non linear function it 
    will be for that particular neaural network
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
                # tanh
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
                ret = x*(1-x)
            else:
                try:
                    ret = 1/(1+np.exp(-x))
                except:
                    ret = 0.0
        elif self._FUNCTION == self._FUNC_TYPES[1]:
            # softmax
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(np.exp(x) / np.sum(np.exp(x)))
            else:
                # from: https://gist.github.com/stober/1946926
                #e_x = np.exp(x - np.max(x))
                #ret = e_x / e_x.sum()
                # from: http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
                ret = np.exp(x) / np.sum(np.exp(x))
        elif self._FUNCTION == self._FUNC_TYPES[2]:
            # relu
            if derivative:
                # from below + http://www.derivative-calculator.net/
                ret = 2*(abs(x))
            else:
                ret = x*(abs(x))
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # tanh
            if derivative:
                # from my own memory of calculus :P
                ret = 1.0-x**2
            else:
                ret = np.tanh(x)
        elif self._FUNCTION == self._FUNC_TYPES[3]:
            # softmax
            if derivative:
                # from wikipedia
                ret = 1.0/(1+np.exp(-x))
            else:
                ret = np.log(1+np.exp(x))
        return ret


class NENonlinear(object):
    """
    Nonlinear
    ---------
    this is used to set up a non linear for a
    network. The idea is you can instantiate it 
    and set what type of non linear function it 
    will be for that particular neaural network
    """
    
    _FUNC_TYPES = ('sigmoid',
                   'softmax',
                   'relu',
                   'tanh',
                   'softplus')
    
    def __init__(self, func_type='sigmoid', bias=1):
        self.bias = bias
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
                # tanh
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
                ret = ne.evaluate('({0}*exp({0}*-x))/(exp({0}*-x)+1)'.format(self.bias))
            else:
                try:
                    ret = ne.evaluate('1/(1+exp({0}*(-x)))'.format(self.bias))
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
            # softmax
            if derivative:
                # from wikipedia
                ret = ne.evaluate('1.0/(1+exp(-x))')
            else:
                ret = ne.evaluate('log(1+exp(x))')
        return ret

class NeuralNetwork(object):
    """
    Neural network
    --------------
    This is my neural netowrk class, it basically holds all 
    my variables and uses my other functions/classes
    """
    def __init__(self, input, hidden, output, non_lin=Nonlinear(), bias=False, alpha=1, ):
        if bias:
            self._BIAS = True
            self._INPUT = input + 1
        else:
            self._BIAS = False
            self._INPUT = input
        self._ALPHA = alpha
        self._HIDDEN = hidden
        self._OUTPUT = output
        self.non_lin = non_lin
        self._init_nodes()

    def _init_nodes(self):
        # set up weights (synapses)
        self.w_in = np.random.randn(self._INPUT, self._HIDDEN) 
        self.w_out = np.random.randn(self._HIDDEN, self._OUTPUT)
        # set up changes
        #self.change_in = np.zeros((self._INPUT, self._HIDDEN))
        #self.change_out = np.zeros((self._HIDDEN, self._OUTPUT))        
        
    def _do_layer(self, layer_in, weights):
        """Does the actual calcs between layers :)"""
        ret = self.non_lin(np.dot(layer_in, weights))
        return ret
    
    #def _error_delta(self, layer_in, y):
        #layer_error = y - layer_in
        #layer_delta = layer_error * self.non_lin(derivative=True)
        #return layer_error, layer_delta
        
    def train(self, x, y, train_loops=1000):
        for i in range(train_loops):

            # from: https://iamtrask.github.io/2015/07/28/dropout/
            
            # Why Dropout: Dropout helps prevent weights from converging to 
            # identical positions. It does this by randomly turning nodes off 
            # when forward propagating. It then back-propagates with all the 
            # nodes turned on.
            # A good initial configuration for this for hidden layers is 50%. 
            # If applying dropout to an input layer, it's best to not exceed 
            # 25%.
            # use Dropout during training. Do not use it at runtime or on your 
            # testing dataset.
            
            #if do_dropout:
                #layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],
                                          #1-dropout_percent)[0] * \
                                          #(1.0/(1-dropout_percent))
            
            # set up layers
            layer0 = x
            layer1 = self._do_layer(layer0, 
                                    self.w_in)
            layer2 = self._do_layer(layer1,
                                    self.w_out)
            
            # calculate errors
            layer2_error = y - layer2
            layer2_delta = layer2_error * self.non_lin(layer2, derivative=True)
            
            layer1_error = layer2_delta.dot(self.w_out.T)
            layer1_delta = layer1_error * self.non_lin(layer1, derivative=True)
            
            if (i % (train_loops/10)) == 0:
                print("loop: {}".format(i))
                print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                print("Guess: ")
                print(layer2[0])               
                print("Guess (round): ")
                print(np.round(layer2[0], 1))
                print("Actual: ")
                print(y[0])
                
            #if (i % (train_loops/100)) == 0:
                #print("currently on loop: {}".format(i))
            # backpropagate error
            self.w_out += self._ALPHA * layer1.T.dot(layer2_delta)
            self.w_in += self._ALPHA * layer0.T.dot(layer1_delta)
            
    
    def guess(self, x):
        _in = x
        _hidden = self.non_lin(np.dot(_in, self.w_in))
        _out = self.non_lin(np.dot(_hidden, self.w_out))
        return _out


class NNN(object):
    """N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
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
            output_error = y - last_layer
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
                

            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                #print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                #print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                #print("Guess: ")
                #print(last_layer[0])
                #print("output delta: ")
                #print(np.round(output_delta, 2))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 1))
                print("Actual: ")
                print(y[0])
        
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
            

class NENNN(object):
    """N-layered neural network"""
    def __init__(self, inputs, weights, outputs, alpha):
        self.trained_loops = 0
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
            self.WEIGHT_DATA[i] = (1/np.sqrt(_in)) * np.random.randn(_in, _out)
            self.LAYER_FUNC[i] = _nonlin
    
    def reset(self):
        self._init_layers()
    
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
            self.trained_loops += 1
            
            # output important info
            if (j % (train_loops/10)) == 0:
                print("loop: {}".format(j))
                #print("Layer1 Error: {}".format(np.mean(np.abs(layer1_error))))                
                #print("Layer2 Error: {}".format(np.mean(np.abs(layer2_error))))
                print("Guess: ")
                print(last_layer[0])
                #print("output delta: ")
                #print(np.round(output_delta, 2))
                print("Guess (rounded): ")
                print(np.round(last_layer[0], 1))
                print("Actual: ")
                print(y[0])
        
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

def serialise(name, nn, path='archive'):
    # check if folder exists and set it up
    create_archive(os.path.join(__file__, path))
    _dir = os.path.join(__file__, path, name)
    if create_archive(_dir):
        pass
    
    # serialise this shizz
    t_x = name + '_tx'
    t_y = name + '_ty'
    t_loop = name + '_tloop'
    hyper = name + '_hyper'
    w_in = name + 'w_in'
    w_out = name + 'w_out'


def deserialise(name):
    pass
    
def create_archive(path='archive'):
    if os.path.exists(os.path.join(__file__, path)):
        return True
    else:
        os.mkdir(os.path.join(__file__, path))
        return False

def make_mnist_np():
    l_items = [100, 500, 1000, 2000, 5000, 10000, 20000]
    for items in l_items:
        print("grabbing {} data...".format(items))
        t_in, t_out = mnist.get_flat_mnist(items=items, normalize=True)
        print("  got nmist array!")
        print('  {}x{}'.format(len(t_in), len(t_in[0])))
        x = np.array(t_in, dtype=np.float)
        y = np.array(t_out, dtype=np.float)
        with open('mnist/tx{}'.format(items), 'wb+') as f:
            pickle.dump(x, f)
        with open('mnist/ty{}'.format(items), 'wb+') as f:
            pickle.dump(y, f)

if __name__ == '__main__':
    
    if True: # using to code fold this shit outa the way
    #x = np.array([[0,1,0],
                  #[3,2,1],
                  #[1,1,1],
                  #[2,1,2]])
    
    #y = np.array([[1],
                  #[1],
                  #[0],
                  #[0]])
    
    #_x = np.array([[1,2,1],
                   #[2,2,2],
                   #[3,1,3]])

    #_y = np.array([[1],
                   #[1],
                   #[0]])    
    
    #------------------------------------------
    # TEST INPUT NODES VS HIDDEN NODES
    # ````````````````````````````````
    # showed for given data set that 2 more hidden
    # node than input node is the most accurate for
    # given data
    
    #for i in range(10):
        #print(str(i) + " more hidden nodes that input:")
        #nn = NeuralNetwork(3, 3+i, 1)
        #nn.train(x, y, 100000)
        #print("  Guess: ")
        #guess = nn.guess(_x)
        #print(guess)
        #print("  Error: ")
        #print(_y - guess)
    
    #-------------------------------------------
    
    #-------------------------------------------
    # TEST TRAIN LOOPS
    # ````````````````
    # 
    
    #for i in range(100000, 1000000, 100000):
        #print(str(i) + " training loops:")
        #nn = NeuralNetwork(3, 4, 1)
        #nn.train(x, y, i)
        #guess = nn.guess(_x)
        #print("  Guess: ")
        #print(guess)
        #print("  Error: ")
        #print(_y - guess)
        
    
    ##-------------------------------------------
    
    ##-------------------------------------------
    ## CVRA calcs with neural network
    ## ``````````````````````````````
    
    ## set up hyper vars
    #temp_x = []
    #temp_y = []
    #import csv
    #with open('', 'rb') as f:
        #csv_file = csv.reader(f)
        #for row in csv:
            #temp_row = []
            #for item in row:
                #temp_row.append(item)
            #x.append(temp_row)
    #x = temp_x
    #with open('', 'rb') as f:
        #csv_file = csv.reader(f)
        #for row in csv:
            #temp_row = []
            #for item in row:
                #temp_row.append(item)
            #x.append(temp_row)
                    
    ## use 1.5 times the inputs for the hidden layer nodes
    #i_input = len(x[0])
    #i_hidden = np.floor(inputs/2) + inputs
    #i_output = 1 # between 0 and 30 hopfully..
    
    #nn = NeuralNetwork(i_input, i_hidden, i_output)
    ## train network
        
    
    ##-------------------------------------------
    ## Diabetes stuff with neural network
    ## ``````````````````````````````````
    
    #import csv
    #t_in = [[]]
    #with open("c:/temp/diabetic_clients_val.tsv", 'rb') as f:
        #csv_r = csv.reader(f, delimiter="\t")
        #rows = []
        #for row in csv_r:
            #rows.append(row)
        #t_in = np.array(rows, dtype=np.float96)
    
    #t_out = [[]]
    #with open("c:/temp/diabetic_clients_stat.tsv", 'rb') as f:
        #csv_r = csv.reader(f, delimiter="\t")
        #rows = []
        #for row in csv_r:
            #rows.append(row)
        #t_out = np.array(rows, dtype=np.float96)

    ##if len(t_in) == len(t_out):
        ##print('length t_in matches t_out')

    #ranges = [(1,150), (0,1), (0,50), (50,200), (25,150), 
              #(0,15), (0,10), (0,20)]
    
    #normalized = normalize(t_in, ranges)
    
    #i_input = len(normalized[0])
    #i_hidden = 1024
    #i_output = 1 # between 0 and 30 hopfully..
    
    ## initialise and train network
    #nn = NeuralNetwork(i_input, i_hidden, i_output, alpha=0.1)    
    #nn.train(np.array(normalized, dtype=np.float96), t_out, 1000)
    
    ## test network
    #x = np.array([41, 1, 35, 94, 58, 4.9, 3.1, 5.3])
    #derp = nn.guess(x)
    
    ## should be 1
    #print(derp)

        



    #-------------------------------------------
    # MNIST attempt 1
    # ```````````````
    
    # neural network with 784 inputs, for flattened input of
    # mnist data. first hidden layer will use sigmoid function 
    # and second hidden layer will be using softmax? 
    
    #import mnist
    
    ## get data
    #print("grabbing data...")
    #t_in, t_out = mnist.get_flat_mnist(items=100, normalize=True)
    #print("  got nmist array!")
    #print('  {}x{}'.format(len(t_in), len(t_in[0])))
    #x = np.array(t_in, dtype=np.float)
    #y = np.array(t_out, dtype=np.float)
    
    
    ## set hypervariables
    #i_input = 784 # this is how many pixel per image (they are flat)
    #i_hidden = 1024
    #i_out = 10
    
    ## initialise network
    #print("initialising network...")
    #nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
    #print("  network initialised!")
    
    ## train network
    #loops = 100
    #print("training network for {} loops".format(loops))
    #nn.train(x, y, loops)
    
    #import pickle
    #derp = pickle.dumps(nn)
    
    #nnn = pickle.loads(derp)
    
    #print("unpickled...")
    #nnn.train(x, y, 100)




    #x = np.array([[0,1,0],
                  #[3,1,1],
                  #[1,0,1],
                  #[0,0,0]])

    #y = np.array([[1],
                  #[1],
                  #[0],
                  #[0]])

    #_x = np.array([[1,0,2],
                   #[1,1,2]])

    #_y = np.array([[0],
                   #[1]])

        pass
    #-------------------------------------------
    # MNIST attempt 2
    # ```````````````
    
    # neural network with 784 inputs, for flattened input of
    # mnist data.
    
    # need to do:
    #  * add a more menu like system to make this into a machine
    #    learning tool!
    
    
    #import mnist
    #try:
        #import cPickle as pickle
    #except:
        #import pickle


    ## get data
    #load_data = input("load mnist training data?")
    #if load_data.lower() == 'y':
        #load_d = input("  enter filename (eg. 500 = tx-500, ty-500): ")
        #with open("mnist/tx{}".format(load_d), 'rb') as f:
            #x = pickle.load(f)
        #with open("mnist/ty{}".format(load_d), 'rb') as f:
            #y = pickle.load(f)
    #else:
        #print("grabbing data...")
        #t_in, t_out = mnist.get_flat_mnist(items=1000, normalize=True)
        #print("  got nmist array!")
        #print('  {}x{}'.format(len(t_in), len(t_in[0])))
        #x = np.array(t_in, dtype=np.float)
        #y = np.array(t_out, dtype=np.float)        
    
    
    #load = input("load network? (y/N): ")
    #if load.lower() == 'y':
        #fname = input("network filename: ")
        #with open(fname, 'rb') as f:
            #nnn = pickle.load(f)
    #else:
        ## set hypervariables
        #i_input = 784 # this is how many pixel per image (they are flat)
        #i_out = 10
        ##even shrink
        ##weights = ((784, 512, NENonlinear('sigmoid')), 
                   ##(512, 256, NENonlinear('sigmoid')),
                   ##(256, 128, NENonlinear('sigmoid')),
                   ##(128, 64, NENonlinear('sigmoid')),
                   ##(64, 32, NENonlinear('sigmoid')),
                   ##(32, 16, NENonlinear('sigmoid')),
                   ##(16, 10, NENonlinear('sigmoid')))
    
        ## less layers
        #weights = ((784, 512, NENonlinear('sigmoid')), 
                   #(512, 256, NENonlinear('sigmoid')),
                   #(256, 16, NENonlinear('sigmoid')),
                   #(16, 10, NENonlinear('sigmoid')))
    
        
        ## initialise network
        #print("initialising network...")
        ##nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
        #nnn = NENNN(inputs=i_input, 
                    #weights=weights, 
                    #outputs=i_out,
                    #alpha=0.01)
        #print("  network initialised!")
    
    ## train networkn
    #loops = 100
    #print("training network for {} loops".format(loops))
    #nnn.train(x, y, loops)
    
    #save = input("save network? (y/N): ")
    #if save.lower() == 'y':
        #fname = input("save network as: ")
        #with open(fname, 'wb+') as f:
            #pickle.dump(nnn, f)
            
            
      
      
    import mnist    
      
    print("grabbing data...")
    t_in, t_out = mnist.get_flat_mnist(items=1000, normalize=True)
    print("  got nmist array!")
    print('  {}x{}'.format(len(t_in), len(t_in[0])))
    x = np.array(t_in, dtype=np.float)
    y = np.array(t_out, dtype=np.float)
    
    
    # set hypervariables
    i_input = 784 # this is how many pixel per image (they are flat)
    i_out = 10
    #even shrink
    #weights = ((784, 512, NENonlinear('sigmoid')), 
               #(512, 256, NENonlinear('sigmoid')),
               #(256, 128, NENonlinear('sigmoid')),
               #(128, 64, NENonlinear('sigmoid')),
               #(64, 32, NENonlinear('sigmoid')),
               #(32, 16, NENonlinear('sigmoid')),
               #(16, 10, NENonlinear('sigmoid')))

    # less layers
    weights = ((784, 32, NENonlinear('sigmoid', 0.1)), 
               #(32, 10, NENonlinear('sigmoid', 0.5)),
               #(32, 16, NENonlinear('sigmoid', 1)),
               (32, 10, NENonlinear('sigmoid', 0.1)))

    
    # initialise network
    print("initialising network...")
    #nn = NeuralNetwork(i_input, i_hidden, i_out, Nonlinear('sigmoid'), False, 0.1)
    nnn = NENNN(inputs=i_input, 
                weights=weights, 
                outputs=i_out,
                alpha=1)
    print("  network initialised!")


    # train network
    loops = 1000

    print("training network for {} loops, with {} alpha".format(loops, nnn._ALPHA))
    nnn.train(x, y, loops)
    
    
    
    #nnn._ALPHA = 0.01
    #print("training network for {} loops, with {} alpha".format(loops, nnn._ALPHA))
    #nnn.train(x, y, loops)
