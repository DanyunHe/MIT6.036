import numpy as np

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return rv(value_list).T

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

def argmax_with_val(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score and the score
    """
    best = l[0]; bestScore = f(best)
    for x in l:
        xScore = f(x)
        if xScore > bestScore:
            best, bestScore = x, xScore
    return (best, bestScore)

def quadratic_linear_gradient(p, y):
    return p - y

def softmax(z):
    epsilon = 1e-10
    v = np.exp(z) + epsilon
    sumv = np.sum(v, axis = 0)
    return v / sumv

def NLL_softmax_gradient(p, y):
    return p - y

def NLL(p, y):
    epsilon = 1e-10
    return float(np.sum(-y * np.log(p + epsilon)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Where s is the output
def sigmoid_gradient(s):
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

# Where th is the output
def tanh_gradient(th):
    return 1 - th**2

def quadratic_loss(y_pred, y):
    return 0.5*np.sum(np.square(y_pred - y))

def quadratic_loss_gradient(y_pred, y):
    return y_pred - y
