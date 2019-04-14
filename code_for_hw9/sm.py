from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        n=len(input_seq)
        s=0
        output=[]
        for i in range(n):
            st=self.transition_fn(s,input_seq[i])
            yt=self.output_fn(st)
            output.append(yt)
            s=st
            
        return output
        
  

class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s



class Binary_Addition(SM):
    start_state = None # Change

    def transition_fn(self, s, x):
        # Your code here
        s=s[0]
        for ele in x:
           s+=ele
            
        return [s//2,s%2]

    def output_fn(self, s):
        # Your code here
        return s[1]
        
    def transduce(self, input_seq):
        n=len(input_seq)
        s=[0,0]
        output=[]
        for i in range(n):
            st=self.transition_fn(s,input_seq[i])
            yt=self.output_fn(st)
            output.append(yt)
            s=st
            
        return output



class Reverser(SM):
    start_state = [None,[]] # Change

    def transition_fn(self, s, x):
        if s[0] is None and x!='end':
            s=[s[0],[x]+s[1]]
        
        elif x=='end':
            s=s[1]
        
        elif s[1:]:
            s=s[1:]
        else:
            s=[None,[]]
        
        print(s,x)
        return s
            
        
    def output_fn(self, s):
        # Your code here
        print(s[0])
        return s[0]
        
    def transduce(self, input_seq):
        n=len(input_seq)
        s=[None,[]]
        output=[]
        for i in range(n):
            st=self.transition_fn(s,input_seq[i])
            yt=self.output_fn(st)
            output.append(yt)
            s=st
            
        return output

            
class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
    def transition_fn(self, s, x):
        return self.f1(self.Wss@s+self.Wsx@x+self.Wss_0)
        
    def output_fn(self, s):
        # Your code here
        return self.f2(self.Wo@s+self.Wo_0)
      
    def transduce(self, input_seq):
        n=len(input_seq)
        s=np.array([[0],[0],[0]])
        output=[]
        for i in range(n):
            st=self.transition_fn(s,input_seq[i])
            yt=self.output_fn(st)
            output.append(yt)
            s=st
          
        return output


