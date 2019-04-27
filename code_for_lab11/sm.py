import numpy as np
from util import softmax

# Give them the class, ask them to implement transduce
class SM:
    start_state = None
    def transition_fn(self, s, i):
        raise NotImplementedError
    def output_fn(self, s):
        raise NotImplementedError

    def transduce(self, input_seq):
        state = self.start_state
        output = []
        for inp in input_seq:
            state = self.transition_fn(state, inp)
            output.append(self.output_fn(state))
        return output

# I'll use this as an example.  Too easy for homework
class Accumulator(SM):
    start_state = 0
    def transition_fn(self, s, i):
        return s + i
    def output_fn(self, s):
        return s

class Binary_Long_Addition(SM):
    start_state = (0, 0)
    def transition_fn(self, s, i):
        (carry, digit) = s
        (i0, i1) = i
        total = i0 + i1 + carry
        return 1 if total > 1 else 0, total % 2

    def output_fn(self, s):
        (carry, digit) = s
        return digit

class Reverser(SM):
    start_state = ([], 'input')

    def transition_fn(self, s, i):
        (symbols, mode) = s
        if i == 'end':
            return symbols, 'output'
        elif mode == 'input':
            return symbols + [i], mode
        else:
            return symbols[:-1], mode

    def output_fn(self, s):
        (symbols, mode) = s
        if mode == 'output' and len(symbols) > 0:
            return symbols[-1]
        else:
            return None

class RNN(SM):
    # Inputs are l x 1
    # States are m x 1
    # Outputs are n x 1
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1 = np.tanh, f2 = softmax):
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.l = self.Wsx.shape[1]
        self.m = self.Wss.shape[0]
        self.n = self.Wo.shape[0]
        # add assertions about shapes matching
        self.start_state = np.zeros((self.m, 1))

        self.f1 = f1
        self.f2 = f2

    def transition_fn(self, s, i):
        return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, i) + self.Wss_0)

    def output_fn(self, s):
        return self.f2(np.dot(self.Wo, s) + self.Wo_0)


# Output 1 if cumulative sum is positive, -1 otherwise
# Offsets are 0, scale output up so tanh is near +1, -1
# Accumulator using RNN structure
acc_RNN = RNN(np.array([[1]]),
              np.array([[1]]),
              np.array([[1000]]),
              np.array([[0]]),
              np.array([[0]]),
              lambda x: x,
              np.tanh)

def t1(input_seq = (1, 0, 1, 1, 4, 3, -2)):
    return Accumulator().transduce(input_seq)

def t2():
    # lowest order digit first
    n1 = (1, 0, 1, 0, 1)
    n2 = (0, 1, 1, 0, 1)
    return Binary_Long_Addition().transduce(zip(n1, n2))

def t3():
    input_seq = ('a', 'b', 'c', 1, 2, 3, 'woo', 'end', 0, 0, 0, 0, 0, 0, 0,
                 0, 0)
    return Reverser().transduce(input_seq)

def t4():
    input_seq = [np.array([[x]]) for x in (1, 2, -4, -4, 10, 1, 1, -20)]
    return acc_RNN.transduce(input_seq)

# Question 2 from extra_final_practice Fall 2017
def t5():
    rnn = RNN(np.array([[1, 0, 0]]).T,
              np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]]),
              np.array([[1, -2, 3]]),
              np.array([[0, 0, 0]]).T,
              np.array([[0]]),
              lambda x: x,
              lambda x: x)
    return rnn.transduce([np.array([[x]]) for x in range(10)])

#print(t5())



