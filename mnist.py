import os
import struct
import numpy as np

# from: https://gist.github.com/akesling/5358964

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


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
    
    
def get_flat_mnist(dataset="training", path="./mnist", items=50000, normalize=False):
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
        flat_image = []
        for row in image:
            l_row = list(row)
            for item in l_row:
                if normalize:
                    if item <= 127:
                        flat_image.append(0)
                    else:
                        flat_image.append(1)
                else:
                    flat_image.append(item)
        flat_images += (flat_image,)
    del images


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


#if __name__ == '__main__':
    #images = tuple()
    #labels = tuple()
    #for image in read():
        #images += (image[1],)
        #labels += (image[0],)
    
    #flat_images = tuple()
    #for image in images:
        #flat_image = []
        #for row in image:
            #l_row = list(row)
            #for item in l_row:
                #flat_image.append(item)
        #flat_images += (flat_image,)
        
    #print(len(flat_images[0]))
