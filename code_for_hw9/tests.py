# Unit tests for questions 1 and 5
import numpy as np
from dist import *
from sm import *
from util import *
from mdp import *

###################################################
# UTILS
###################################################
def softmax(z):
    v = np.exp(z)
    return v / np.sum(v, axis = 0)


def tiny_reward(s, a):
    # Tiny MDP reward function
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0


def tiny_transition(s, a):
    # Tiny MDP transition function
    if s == 0:
        if a == 'b':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.9})

###################################################
# QUESTION 1 UNIT TESTS
###################################################

def test_accumulator_sm():
    res = Accumulator().transduce([-1, 2, 3, -2, 5, 6])
    assert(res == [-1, 1, 4, 2, 7, 13])
    print("Test passed!")


def test_binary_addition_sm():
    res = Binary_Addition().transduce([(1, 1), (1, 0), (0, 0)])
    assert(res == [0, 0, 1])
    print("Test passed!")


def test_reverser_sm():
    res = Reverser().transduce(['foo', ' ', 'bar'] + ['end'] + list(range(5)))
    assert(res == [None, None, None, 'bar', ' ', 'foo', None, None, None])
    print("Test passed!")


def test_rnn():
    Wsx1 = np.array([[0.1],
                     [0.3],
                     [0.5]])
    Wss1 = np.array([[0.1,0.2,0.3],
                     [0.4,0.5,0.6],
                     [0.7,0.8,0.9]])
    Wo1 = np.array([[0.1,0.2,0.3],
                    [0.4,0.5,0.6]])
    Wss1_0 = np.array([[0.01],
                       [0.02],
                       [0.03]])
    Wo1_0 = np.array([[0.1],
                      [0.2]])
    in1 = [np.array([[0.1]]),
           np.array([[0.3]]),
           np.array([[0.5]])]

    rnn = RNN(Wsx1, Wss1, Wo1, Wss1_0, Wo1_0, np.tanh, softmax)
    expected = np.array([[[0.4638293846951024], [0.5361706153048975]],
                        [[0.4333239107898491], [0.566676089210151]],
                        [[0.3821688606165438], [0.6178311393834561]]])

    assert(np.allclose(expected, rnn.transduce(in1)))
    print("Test passed!")


def test_acc_sign_rnn(acc_sign_rnn):
    res = acc_sign_rnn.transduce([-1, -2, 2, 3, -3, 1])
    expected = np.array([[[-1.0]], [[-1.0]], [[-1.0]], [[1.0]], [[-1.0]], [[0.0]]])
    assert(np.allclose(expected, res))
    print("Test passed!")


def test_auto_rnn(auto_rnn):
    res = auto_rnn.transduce([np.array([[x]]) for x in range(-2,5)])
    expected = np.array([[[-2.0]], [[3.0]], [[-4.0]], [[-2.0]], [[0.0]], [[2.0]], [[4.0]]])
    assert(np.allclose(expected, res))
    print("Test passed!")

###################################################
# QUESTION 5 UNIT TESTS
###################################################

def test_value():
    q = TabularQ([0,1,2,3], ['b','c'])
    q.set(0, 'b', 5)
    q.set(0, 'c', 10)
    assert(value(q, 0) == 10)
    print("Test passed!")


def test_greedy():
    q = TabularQ([0, 1, 2, 3],['b', 'c'])
    q.set(0, 'b', 5)
    q.set(0, 'c', 10)
    q.set(1, 'b', 2)
    assert(greedy(q, 0) == 'c')
    assert(greedy(q, 1) == 'b')
    print("Test passed!")


def test_epsilon_greedy():
    q = TabularQ([0, 1, 2, 3],['b', 'c'])
    q.set(0, 'b', 5)
    q.set(0, 'c', 10)
    q.set(1, 'b', 2)
    eps = 0.0
    assert(epsilon_greedy(q, 0, eps) == 'c')
    assert(epsilon_greedy(q, 1, eps) == 'b')
    print("Test passed!")


def test_value_iteration():
    tiny = MDP([0, 1, 2, 3], ['b', 'c'], tiny_transition, tiny_reward, 0.9)
    q = TabularQ(tiny.states, tiny.actions)
    qvi = value_iteration(tiny, q, eps=0.1, max_iters=100)
    expected = dict([((2, 'b'), 5.962924188028282),
                     ((1, 'c'), 5.6957634856549095),
                     ((1, 'b'), 5.6957634856549095),
                     ((0, 'b'), 5.072814297918393),
                     ((0, 'c'), 5.262109602844769),
                     ((3, 'b'), 6.794664584556008),
                     ((3, 'c'), 6.794664584556008),
                     ((2, 'c'), 5.962924188028282)])
    for k in qvi.q:
        assert((qvi.q[k] - expected[k]) < 1.0e-5)
    print("Test passed!")


def test_q_em():
    tiny = MDP([0, 1, 2, 3], ['b', 'c'], tiny_transition, tiny_reward, 0.9)
    assert(np.allclose([q_em(tiny, 0, 'b', 1)], [0.0]))
    assert(np.allclose([q_em(tiny, 0, 'b', 2)], [0.81]))
    assert(np.allclose([q_em(tiny, 0, 'b', 3)], [1.0287000000000002]))
    assert(np.allclose([q_em(tiny, 0, 'c', 3)], [1.4103]))
    assert(np.allclose([q_em(tiny, 2, 'b', 3)], [1.9116000000000002]))
    print("Tests passed!")

###################################################
# Call your tests below
###################################################
test_accumulator_sm()
test_binary_addition_sm()
test_reverser_sm()
test_rnn()

test_value()
test_greedy()
test_epsilon_greedy()
test_value_iteration()
test_q_em()
