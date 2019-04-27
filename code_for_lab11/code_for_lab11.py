import pdb
from util import *
import numpy as np
import sm
import functools
import random
import _pickle as cPickle

############################################################################
#
#  Testing RNN Learning
#
############################################################################

# np.random.seed(0) # set the random seed to ensure same output
np.seterr(over='raise')

# Learn accumulator
def test_linear_accumulator(num_steps = 10000,
                            num_seqs = 100, seq_length = 5,
                            step_size = .01):
    # generate random training data: num_seqs of seq_length of random
    # numbers between -0.5 and 0.5.
    data = []
    for _ in range(num_seqs):           
        x = np.random.random((1, seq_length)) - 0.5 # seq in
        y = np.zeros((1, seq_length))               # seq out
        for j in range(seq_length):
            y[0, j] = x[0, j] + (0.0 if j == 0 else y[0, j-1])
        data.append((x, y))
    # specify rnn
    rnn = RNN(1, 1, 1, quadratic_loss, lambda z: z, quadratic_linear_gradient,
              step_size, lambda z: z, lambda z: 1)
    # train it
    rnn.train_seq_to_seq(data, num_steps)
    # print weights
    print("\nWeights:", flush=True)
    print("Wsx:", rnn.Wsx); print("Wss:", rnn.Wss); print("Wo:", rnn.Wo)
    print("Wss0:", rnn.Wss0); print("Wo0:", rnn.Wo0)
    return rnn

# Learn delay of 1 dimensional data, e.g. numbers
def test_delay_num(delay = 1, num_steps = 10000,
               num_seqs = 10000, seq_length = 10,
               step_size = .005):
    # generate random training data: num_seqs of seq_length of random
    # numbers (between 0 and 1)
    data = []
    for _ in range(num_seqs):
        vals = np.random.random((1, seq_length))
        x = np.hstack([vals, np.zeros((1, delay))]) # seq in, pad right
        y = np.hstack([np.zeros((1, delay)), vals]) # seq out, pad left
        data.append((x, y))
    # specify rnn
    m = (delay + 1) * 2
    rnn = RNN(1, m, 1, quadratic_loss, lambda z: z, quadratic_linear_gradient,
              step_size, lambda z: z, lambda z: 1)
    # train it
    rnn.train_seq_to_seq(data, num_steps)
    # print weights
    print("\nWeights:", flush=True)
    print("Wsx:", rnn.Wsx); print("Wss:", rnn.Wss); print("Wo:", rnn.Wo)
    print("Wss0:", rnn.Wss0); print("Wo0:", rnn.Wo0)
    # construct a state machine and test with a (fixed) sequence
    mm = rnn.sm()
    print(mm.transduce([np.array([[float(v)]]) \
                        for v in [3, 4, 5, 1, 2, -1, 6]]))
    return rnn

# Learn delay of one-hot encoided characters, e.g. numbers
def test_delay_char(delay = 1, num_steps = 10000,
                    alphabet = tuple(range(10)),
                    num_seqs = 10000, seq_length = 4, step_size = .001):
    # generate random training data: num_seqs of seq_length of
    # integers, represented by one-hot vectors over the alphabet.
    codec = OneHotCodec(alphabet)
    n = codec.n
    data = []
    for _ in range(num_seqs):
        rand_seq = np.random.random_integers(0, n-1, seq_length)
        vals = codec.encode_seq(rand_seq) # one-hot typically
        # first word of alphabet is start/end symbol
        pad = codec.encode_seq(alphabet[0:1] * delay)
        x = np.hstack([vals, pad])
        y = np.hstack([pad, vals])
        data.append((x, y))
    # specify rnn
    m = (delay + 1) * n
    f1, df1 = tanh, tanh_gradient
    loss, f2, dLdf2 = NLL, softmax, NLL_softmax_gradient
    rnn =  RNN(n, m, n, loss, f2, dLdf2,
               step_size, f1, df1)
    # train it
    rnn.train_seq_to_seq(data, num_steps)
    # construct a state machine and test with a (fixed) sequence
    mm = rnn.sm()
    vin = [codec.encode(c) for c in [0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 1]]
    vout = mm.transduce(vin)
    cout = [codec.decode_max(v) for v in vout]
    print(cout)
    return rnn

# interpret first bit as lowest order;  may leave off highest-order bit
# s1 and s2 are 1 x k
# return 1 x k
def bin_add(s1, s2):                    # binary add
    k = s1.shape[1]
    result = np.zeros((1, k))
    carry = 0
    for j in range(k):
        tot = s1[0, j] + s2[0, j] + carry
        result[0,j] = tot % 2
        carry = 1 if tot > 1 else 0
    return result

# Learn binary addition
def test_binary_addition(num_seqs = 1000, seq_length = 5, num_steps = 50000,
                         step_size = 0.01, num_hidden = 8):
    # generate random training data: num_seqs of seq_length of
    # binary integers.
    data = []
    for _ in range(num_seqs):
        s1 = np.random.random_integers(0, 1, (1, seq_length))
        s2 = np.random.random_integers(0, 1, (1, seq_length))
        x = np.vstack([s1, s2])         # seq in
        y = bin_add(s1, s2)             # seq out
        data.append((x, y))
    # specify rnn
    l = 2 # two input dimensions
    m = num_hidden
    n = 1 # one output dimension
    f1 = sigmoid; df1 = sigmoid_gradient
    loss = quadratic_loss
    f2 = lambda z: z; dldz2 = quadratic_linear_gradient
    rnn =  RNN(l, m, n, loss, f2, dldz2, step_size, f1, df1)
    # train it
    rnn.train_seq_to_seq(data, num_steps)
    # construct a state machine and test with a (fixed) sequence
    mm = rnn.sm()
    n1 = '01101'
    n2 = '01111'
    # answer is:    11100
    a = [np.array([[float(d1), float(d2)]]).T for d1, d2 in zip(n1, n2)]
    vin = list(reversed(a))
    vout = mm.transduce(vin)
    print(' in:', vin)
    print('out:', vout)
    return rnn

# Predicting the next character in a sequence

# Generate the training data for predicting next character
def process_seq_data(words):
    alphabet = sorted(list(functools.reduce(lambda a, b: set(a) | set(b),
                                            words, set())))
    codec = OneHotCodec(alphabet + ['.'])
    data = []
    for w in words:
        vals = codec.encode_seq(w)
        pad = codec.encode_seq(['.'])
        y = np.hstack([vals, pad])
        x = np.hstack([pad, vals])
        data.append((x, y))
    return data, codec

# Train rnn
def train_seq(data, codec, num_steps = 10000, step_size = 0.01,
              num_hidden = 10, interactive = False):
    # specify rnn
    l = codec.n
    m = num_hidden
    n = codec.n
    f1 = tanh; df1 = tanh_gradient
    loss = NLL
    f2 = softmax; dldz2 = NLL_softmax_gradient
    rnn =  RNN(l, m, n, loss, f2, dldz2, step_size, f1, df1)
    # train it
    rnn.train_seq_to_seq(data, num_steps)
    return rnn

# Generate the training data and carry out the prediction
def generate_seq(words, num_steps = 10000, step_size = 0.05, num_hidden = 20,
              split=0., interactive = False, interactive_top5 = False):
    data, codec = process_seq_data(words)    
    if split and len(words) > 1:
        random.shuffle(data)
        k = int((1-split)*len(data))
        data1, data2 = data[:k], data[k:]
        print('Training set size', len(data1), 'Held out set size', len(data2))
    else:
        data1, data2 = data, None
    rnn = train_seq(data1, codec, num_steps = num_steps,
                    step_size = step_size, num_hidden = num_hidden,
                    interactive = interactive)
    if data2:
        total_loss = 0
        for (x, y) in data2:
            loss, _, _ = rnn.forward_seq(x,y)
            total_loss += loss
        print('Average loss on held out set:', total_loss/len(data2))
    # test by either generating random strings from the beginning or
    # interactively starting from input provided by user.
    if interactive and not(interactive_top5):
        for _ in range(100):
            rnn.gen_seq_interactive(codec)
    elif interactive_top5:
        rnn.gen_seq_interactive_top5(codec)
    else:
        for _ in range(100):
            print(''.join(rnn.gen_seq('.', 100, codec)))
    return rnn, codec


def save_seq(words, out_file,interactive, num_steps = 20000, step_size = 0.001, num_hidden = 150,
              split=0.):
    data, codec = process_seq_data(words)    
    if split and len(words) > 1:
        random.shuffle(data)
        k = int((1-split)*len(data))
        data1, data2 = data[:k], data[k:]
        print('Training set size', len(data1), 'Held out set size', len(data2))
    else:
        data1, data2 = data, None
    rnn = train_seq(data1, codec, num_steps = num_steps,
                    step_size = step_size, num_hidden = num_hidden,
                    interactive = interactive)
    
    cPickle.dump(rnn, open('models/' + out_file, 'wb'))

def load_seq(words, in_file,  interactive, interactive_top5, num_steps = 20000, step_size = 0.001, num_hidden = 150,
              split=0.):
    rnn = cPickle.load(open(in_file,'rb'))
    data, codec = process_seq_data(words) 
    
    if interactive and not(interactive_top5):
        for _ in range(100):
            rnn.gen_seq_interactive(codec)
    elif interactive_top5:
        rnn.gen_seq_interactive_top5(codec)
    else:
        for _ in range(100):
            print(''.join(rnn.gen_seq('.', 100, codec)))
    return rnn, codec
    
    

############################################################################
#
#  Sequence predictions on word lists
#
############################################################################

long_words = ['alabama', 'arkansas', 'mississippi', 'madagascar', 
              'taradiddle', 'hippopotamus', 'missasolemnis', 'abcdefghij']

dirname = ''                            # default will be current directory

def test_word(word, interactive = False, num_hidden=1, num_steps=10000, step_size=0.005):
    return generate_seq([word], num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)

# Learn models for two classes, then do classification.
def test_language(english = True, split=0.25,
                  num_hidden=300, num_steps=500000, step_size=0.001):
    if english:
        data = read_words(dirname + 'baskervilles.txt')
    else:
        data = read_words(dirname + 'mousquetaires.txt')

    long = [w.lower() for w in data if len(w) > 5]
    print(len(long), 'long english words')

    return generate_seq(long, split=split, num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)

# Generate heavy metal band names
def test_metal(interactive = True, interactive_top5 = True, split=0, num_hidden = 150, num_steps = 20000, step_size = .001, train=True):
    data = read_lines(dirname + 'metal_bands.txt')
    if train==True:
        return generate_seq(data, interactive = interactive, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
    else:
        in_file = 'models/metal_rnn.p'
        return load_seq(data, in_file, interactive = interactive, interactive_top5 = interactive_top5, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
 
# Generate MIT class names
def test_class_names(interactive = True, interactive_top5 = True, split=0, num_hidden = 150, num_steps = 20000, step_size = .001, train=True):
    data = read_lines(dirname + 'MIT_classes.txt')
    if train==True:
        return generate_seq(data, interactive = interactive, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
    else:
        in_file = 'models/MIT_classes_rnn.p'
        return load_seq(data, in_file, interactive = interactive, interactive_top5 = interactive_top5, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)

def test_company_names(interactive = True, interactive_top5 = True, split=0, num_hidden = 150, num_steps = 20000, step_size = .001, train=True):
    data = read_lines(dirname + 'companies.txt')
    if train==True:
        return generate_seq(data, interactive = interactive, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
    else:
        in_file = 'models/companies_rnn.p'
        return load_seq(data, in_file, interactive = interactive, interactive_top5 = interactive_top5, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)

       
# Generate food names
def test_food(interactive = True,interactive_top5 = True,  split=0, num_hidden = 150, num_steps = 20000, step_size  =.001, train=True):
    data = read_lines(dirname + 'food.txt')
    if train==True:
        return generate_seq(data, interactive = interactive, split=split,
                        num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
    else:
        in_file = 'models/food_rnn.p'
        return load_seq(data, in_file, interactive = interactive, interactive_top5 = interactive_top5, split=split,
                            num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)
    
    

# Generate lines from poem
def test_poem(interactive=False, split=0, num_hidden = 80, num_steps = 20000, step_size  = .001):
    data = read_lines(dirname + 'simple_poem.txt')
    return generate_seq(data, interactive = interactive, split=split,
                        num_hidden=num_hidden, num_steps=num_steps, step_size=step_size)

# Utilities for reading files.
def read_words(fileName):
    result = []
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f:
            result.extend(line.split()) # split the words out
    return result

def read_lines(fileName):
    result = []
    with open(fileName, 'r', encoding='utf-8' ) as f:
        for line in f:
            result.append(line)
    return result

############################################################################
#
# RNN class
#
############################################################################

# Based on an implementation by Michael Sun

class RNN:
    weight_scale = .0001
    def __init__(self, input_dim, hidden_dim, output_dim, loss_fn, f2, dloss_f2, step_size=0.1,
                 f1 = tanh, df1 = tanh_gradient, init_state = None,
                 Wsx = None, Wss = None, Wo = None, Wss0 = None, Wo0 = None,
                 adam = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.dloss_f2 = dloss_f2
        self.step_size = step_size
        self.f1 = f1
        self.f2 = f2
        self.df1 = df1
        self.adam = adam
        self.init_state = init_state if init_state is not None else \
                                         np.zeros((self.hidden_dim, 1))
        self.hidden_state = self.init_state
        self.t = 0

        # Initialize weight matrices
        self.Wsx = Wsx if Wsx is not None \
                    else np.random.random((hidden_dim, input_dim)) * self.weight_scale
        self.Wss = Wss if Wss is not None \
                       else np.random.random((hidden_dim, hidden_dim)) * self.weight_scale
        self.Wo = Wo if Wo is not None \
                     else np.random.random((output_dim, hidden_dim)) * self.weight_scale
        self.Wss0 = Wss0 if Wss0 is not None \
                         else np.random.random((hidden_dim, 1)) * self.weight_scale
        self.Wo0 = Wo0 if Wo0 is not None \
                       else np.random.random((output_dim, 1)) * self.weight_scale

        # Initialization for ADAM
        if adam:
            self.dLdWsx_sq = np.zeros_like(self.Wsx)
            self.dLdWo_sq = np.zeros_like(self.Wo)
            self.dLdWss_sq = np.zeros_like(self.Wss)
            self.dLdWo0_sq = np.zeros_like(self.Wo0)
            self.dLdWss0_sq = np.zeros_like(self.Wss0)

            self.dLdWsx_m = np.zeros_like(self.Wsx)
            self.dLdWo_m = np.zeros_like(self.Wo)
            self.dLdWss_m = np.zeros_like(self.Wss)
            self.dLdWo0_m = np.zeros_like(self.Wo0)
            self.dLdWss0_m = np.zeros_like(self.Wss0)


    # Just one step of forward propagation.  x and y are for a single time step
    # Depends on self.hidden_state and reassigns it
    # Returns predicted output, loss on this output, and dLoss_dz2
    def forward_propagation(self, x):
        new_state = self.f1(np.dot(self.Wsx, x) +
                            np.dot(self.Wss, self.hidden_state) + self.Wss0)
        z2 = np.dot(self.Wo, new_state) + self.Wo0
        p = self.f2(z2)
        self.hidden_state = new_state
        return p

    def forward_prop_loss(self, x, y):
        p = self.forward_propagation(x)
        loss = self.loss_fn(p, y)
        dL_dz2 = self.dloss_f2(p, y)
        return p, loss, dL_dz2

    def b(self, xs, dL_dz2, states):
        dC = np.zeros_like(self.Wsx)
        dB = np.zeros_like(self.Wss)
        dA = np.zeros_like(self.Wo)
        dB0 = np.zeros_like(self.Wss0)
        dA0 = np.zeros_like(self.Wo0)
        dLfuture_dst = np.zeros((self.hidden_dim, 1))
        k = xs.shape[1]
        for t in range(k-1, -1, -1):
            xt = xs[:, t:t+1]
            st = states[:, t:t+1]
            st_minus_1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
            dL_dz2t = dL_dz2[:, t:t+1]
            dL_dA = np.dot(dL_dz2t, st.T)
            dL_dA0 = dL_dz2t
            dLtfuture_dst = np.dot(self.Wo.T, dL_dz2t) + dLfuture_dst
            dLtfuture_dz1t = dLtfuture_dst * self.df1(st)
            dLtfuture_dB = np.dot(dLtfuture_dz1t, st_minus_1.T)
            dLtfuture_dB0 = dLtfuture_dz1t
            dLtfuture_dC = np.dot(dLtfuture_dz1t, xt.T)
            dLfuture_dst = np.dot(self.Wss.T, dLtfuture_dz1t)
            dC += dLtfuture_dC
            dB += dLtfuture_dB
            dB0 += dLtfuture_dB0
            dA += dL_dA
            dA0 += dL_dA0
        return dC, dB, dA, dB0, dA0

    # With adagrad
    def sgd_step(self, xs, dLdz2s, states,
                 gamma1 = 0.9, gamma2 = 0.999, fudge = 1.0e-8):
        dWsx, dWss, dWo, dWss0, dWo0 = self.b(xs, dLdz2s, states)

        self.t += 1

        if self.adam:
            self.dLdWsx_m = gamma1 * self.dLdWsx_m + (1 - gamma1) * dWsx
            self.dLdWo_m = gamma1 * self.dLdWo_m + (1 - gamma1) * dWo
            self.dLdWss_m = gamma1 * self.dLdWss_m + (1 - gamma1) * dWss
            self.dLdWo0_m = gamma1 * self.dLdWo0_m + (1 - gamma1) * dWo0
            self.dLdWss0_m = gamma1 * self.dLdWss0_m + (1 - gamma1) * dWss0

            self.dLdWsx_sq = gamma2 * self.dLdWsx_sq + (1 - gamma2) * dWsx ** 2
            self.dLdWo_sq = gamma2 * self.dLdWo_sq + (1 - gamma2) * dWo ** 2
            self.dLdWss_sq = gamma2 * self.dLdWss_sq + (1 - gamma2) * dWss ** 2
            self.dLdWo0_sq = gamma2 * self.dLdWo0_sq + (1 - gamma2) * dWo0 ** 2
            self.dLdWss0_sq = gamma2 * self.dLdWss0_sq + (1 - gamma2) * dWss0 ** 2

            # Correct for bias
            dLdWsx_mh = self.dLdWsx_m / (1 - gamma1**self.t)
            dLdWo_mh = self.dLdWo_m / (1 - gamma1**self.t)
            dLdWss_mh = self.dLdWss_m / (1 - gamma1**self.t)
            dLdWo0_mh = self.dLdWo0_m / (1 - gamma1**self.t)
            dLdWss0_mh = self.dLdWss0_m / (1 - gamma1**self.t)

            dLdWsx_sqh = self.dLdWsx_sq / (1 - gamma2**self.t)
            dLdWo_sqh = self.dLdWo_sq / (1 - gamma2**self.t)
            dLdWss_sqh = self.dLdWss_sq / (1 - gamma2**self.t)
            dLdWo0_sqh =  self.dLdWo0_sq / (1 - gamma2**self.t)
            dLdWss0_sqh =  self.dLdWss0_sq / (1 - gamma2**self.t)

            self.Wsx -= self.step_size * (dLdWsx_mh /
                                          (fudge + np.sqrt(dLdWsx_sqh)))
            self.Wss -= self.step_size * (dLdWss_mh /
                                          (fudge + np.sqrt(dLdWss_sqh)))
            self.Wo -= self.step_size * (dLdWo_mh /
                                         (fudge + np.sqrt(dLdWo_sqh)))
            self.Wss0 -= self.step_size * (dLdWss0_mh /
                                           (fudge + np.sqrt(dLdWss0_sqh)))
            self.Wo0 -= self.step_size * (dLdWo0_mh /
                                          (fudge + np.sqrt(dLdWo0_sqh)))
        else:
            self.Wsx -= self.step_size * dWsx
            self.Wss -= self.step_size * dWss
            self.Wo -= self.step_size * dWo
            self.Wss0 -= self.step_size * dWss0
            self.Wo0 -= self.step_size * dWo0

    def reset_hidden_state(self):
        self.hidden_state = self.init_state

    def forward_seq(self, x, y):
        k = x.shape[1]
        dLdZ2s = np.zeros((self.output_dim, k))
        states = np.zeros((self.hidden_dim, k))
        train_error = 0.0
        self.reset_hidden_state()
        for j in range(k):
            p, loss, dLdZ2 = self.forward_prop_loss(x[:, j:j+1], y[:, j:j+1])
            dLdZ2s[:, j:j+1] = dLdZ2
            states[:, j:j+1] = self.hidden_state
            train_error += loss
        return train_error/k, dLdZ2s, states

    # For now, training_seqs will be a list of pairs of np arrays.
    # First will be l x k second n x k where k is the sequence length
    # and can be different for each pair
    def train_seq_to_seq(self, training_seqs, steps = 100000,
                         print_interval = None):
        if print_interval is None: print_interval = int(steps / 10)
        num_seqs = len(training_seqs)
        total_train_err = 0
        self.t = 0
        iters = 1
        for step in range(steps):
            i = np.random.randint(num_seqs)
            x, y = training_seqs[i]
            avg_seq_train_error, dLdZ2s, states = self.forward_seq(x, y)

            # Check the gradient computation against the numerical grad.
            # grads = self.b(x, dLdZ2s, states)
            # grads_n = self.num_grad(lambda : forward_seq(x, y, dLdZ2s,
            # states)[0])
            # compare_grads(grads, grads_n)

            self.sgd_step(x, dLdZ2s, states)
            total_train_err += avg_seq_train_error
            if (step % print_interval) == 0 and step > 0:
                print('%d/10: training error'%iters, total_train_err / print_interval, flush=True)
                total_train_err = 0
                iters += 1

    def num_grad(self, f, delta=0.001):
        out = []
        for W in (self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0):
            Wn = np.zeros(W.shape)
            out.append(Wn)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    wi = W[i,j]
                    W[i,j] = wi - delta
                    fxm = f()
                    W[i,j] = wi + delta
                    fxp = f()
                    W[i,j] = wi
                    Wn[i,j] = (fxp - fxm)/(2*delta)
        return out

    # Return a state machine made out of these weights
    def sm(self):
        return sm.RNN(self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0,
                      self.f1, self.f2)

    # Assume that input and output are same dimension
    def gen_seq(self, init_sym, seq_len, codec):
        assert self.input_dim == self.output_dim
        assert self.f2 == softmax
        result  = []
        self.reset_hidden_state()
        x = codec.encode(init_sym)
        for _ in range(seq_len):
            p = self.forward_propagation(x)
            x = np.array([np.random.multinomial(1, p.T[0])]).T
            if codec.decode(x) == '.':
                break
            result.append(codec.decode(x))
        return result

    def gen_seq_interactive(self, codec, seq_len = 100, maximize = True):
        self.reset_hidden_state()
        start = '.' + input('Starting string: ')
        result = start[1:]
        for c in start:
            p = self.forward_propagation(codec.encode(c))
        for _ in range(seq_len):
            c = codec.decode_max(p)
            #if c in ['.', '\n', ' ']:  break
            if c in ['.', '\n']:  break
            result = result + c
            x = codec.encode(c)
            p = self.forward_propagation(x)
        print(result)
        return result
    
    def gen_seq_interactive_top5(self, codec, seq_len = 100, maximize = True):
        self.reset_hidden_state()
        start = '.' + input('Starting string: ')
        result = start[1:]
        for c in start:
            p = self.forward_propagation(codec.encode(c))
        
        while True:
            c = codec.decode_max(p)
            #if c in ['.', '\n', ' '] :  break
            if c in ['.', '\n']:  break
            c_top5 = codec.decode_top5(p)
            #print("The argmax is :", c)
            print("We recommend that you type one of the top 5 most frequent alphabets that follow '" + str(result) + "' : ", c_top5)
            next_character = input("Next character after '" + str(result) + "' : ")
            result = result + next_character
            x = codec.encode(next_character)
            p = self.forward_propagation(x)
        print("Your final result is : ", result)
        return result

def compare_grads(g, gn):
    names = ('Wsx', 'Wss', 'Wo', 'Wss0', 'Wo0')
    for i in range(len(g)):
        diff = np.max(np.abs(g[i]-gn[i]))
        if diff > 0.001:
            print('Diff in', names[i], 'is', diff)
            print('Analytical')
            print(g[i])
            print('Numerical')
            print(gn[i])
            input('Go?')

############################################################################
#
# One-hot encoding/decoding
#
############################################################################

class OneHotCodec:
    def __init__(self, alphabet):
        pairs = list(enumerate(alphabet))
        self.n = len(pairs)
        self.coder = dict([(c, i) for (i, c) in pairs])
        self.decoder = dict(pairs)

    # Take a symbol, return a one-hot vector
    def encode(self, c):
        return self.encode_index(self.coder[c])

    # Take an index, return a one-hot vector
    def encode_index(self, i):
        v = np.zeros((self.n, 1))
        v[i, 0] = 1
        return v

    # Take a one-hot vector, return a symbol
    def decode(self, v):
        return self.decoder[int(np.nonzero(v)[0])]

    # Take a probability vector, return max likelihood symbol
    def decode_max(self, v):
        return self.decoder[np.argmax(v)]
    
    def decode_top5(self, v):
        v_viewed = v.reshape(v.shape[0])
        top5_args = np.argsort(v_viewed)[-5:][::-1].tolist()
        return [self.decoder[arg] for arg in top5_args]

    def encode_seq(self, cs):
        return np.hstack([self.encode(c) for c in cs])


if __name__ == "__main__":
    # test_linear_accumulator()
    # test_word("aabaaabbaaaababaabaa", num_hidden=7, num_steps=10000)
    # test_class_names(interactive = True, interactive_top5 = False, num_hidden = 150, num_steps = 20000, step_size = .001)
    #test_metal()
    # test_food(interactive=False, interactive_top5=True,train=False)
    test_company_names(interactive=False, interactive_top5=True,train=False)
    # pass
