
=====

CrossEntropyTest.py

=====

import sys
import numpy as np
import torch

def main():
    a = torch.arange(10)
    a2 = a.view(-1,10)
    print(a2)
    a3 = a2[:,-3:]
    print(a3)
    # assume 3 outputs and batch size of 2, so logits = 2x3 tensor
    logits=torch.tensor([[1,3.0,5],[2,4.0,1]])
    # above indicates predicted output is 2 and 1 (index of highest value)
    print('-----logits--------')
    print(logits)
    targets=torch.tensor([2,0])
    # targets are specified as long, i.e.,
    # index of which output is to be recognized, try with [2,1] to see if loss
    decreases
    # pytorch's cross entropy loss, operates on logits (not on softmax layer)
    loss = torch.nn.functional.cross_entropy(logits,targets)
    print('\ncross entropy loss by pytorch=', loss)
    # pytorch's nll_loss (negative log likelihood loss) is similar to cross
    entropy
    # it operates on log_sofmax, rather than raw logits
    outs = torch.softmax(logits,dim=1)
    print('-----softmax------')
    print(outs)
    outs2 = torch.nn.functional.log_softmax(logits, dim=1)
    loss_nll = torch.nn.functional.nll_loss(outs2,targets)
    print('nll loss by pytorch =',loss_nll)
    # compute cross entropy ourselves
    z = (np.log(outs[0,targets[0]]) + np.log(outs[1,targets[1]]))/2
    print("\ncross entropy by our calculation=",-z)
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

EinopsTest.py

=====

import sys
import torch    
from einops import rearrange
 
def main():
    A = torch.tensor([[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]])
    B = torch.tensor([[1, 2, 1, 1],
    [3, 4, 2, 5],
    [1, 3, 6, 7],
    [1, 4, 6, 8]])
    print(A)
    print(B)
    C = torch.einsum('ij, jk -> ik', A, B) # matrix mult.
    print(C)
    C1 = torch.einsum('ij, jk -> ik', A, B) # matrix mult.
    print(C1)
    C2 = torch.einsum('ij, kj -> ik', A, B) # Ax(transpose(B) - matrix mult.
    print(C2)
    C3 = torch.einsum("ii -> i", A) # diagonal elements only
    print(C3)
    C4 = torch.einsum("ii -> ", A) # sum diagonal elements - trace
    print(C4)
    C5 = torch.einsum("ij -> j", A) # sum column elements (row wise sum)
    print(C5)
    C6 = torch.einsum('ij, ij -> ij', A, B) # element wise product
    print(C6)
    C6b = torch.einsum('ij, ij -> ', A, B) # element wise product THEN ADD ALL
    print(C6b)
    C7 = torch.einsum('ij, ij, ij -> ij', A, A, A) # cube elements
    print(C7)
    C8 = torch.einsum('ij -> ji', A)
    print(C8)
    # transpose
    C9 = torch.einsum('ij,ij -> i', A, B)
    print(C9)
    # multiply row wise and add each row
    d1 = torch.tensor([3, 5, 7, 9])
    d2 = torch.tensor([1, 2, 3, 4])
    douter = torch.einsum('i, j -> ij', d1, d2) # outer product
    print(douter)
    dinner = torch.einsum('i, i -> ', d1, d2) # inner product
    print(dinner)
    dfrobenius = torch.einsum("ij, ij -> ", A, A) # frobenius norm
    # sum of squares of all elements of a matrix
    print('Frobenius norm...')
    print(dfrobenius)
    batch_tensor_1 = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    print(batch_tensor_1)
    batch_tensor_2 = torch.arange(2 * 4 * 3).reshape(2, 3, 4)
    print(batch_tensor_2)
    dmul = torch.einsum('bij, bjk -> bik', batch_tensor_1, batch_tensor_2) #batch matrix multiplication
    print(dmul)
    dt = torch.randn((3,5,4,6,8,2,7,9)) # 8 dimensions
    print(dt.shape)
    esum = torch.einsum("ijklmnop -> p", dt)
    # marginalize or sum over dim p
    print(esum) # produces 9 numbers, try op instead of p
    kv = torch.zeros((2,1024,64)) # 2 is batch size
    q = torch.zeros((2,1024,64))
    q2 = rearrange(q,'b (n s) e->b n s e', s=16)
    print(q2.shape) #[2,64,16,64]
    q3 = rearrange(q2,'b n s e-> (b n) s e')
    print(q3.shape) #[128,16,64]
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

KLDivergenceTest.py

=====

import numpy as np
import sys

def kl(p, q):
    #Kullback-Leibler divergence D(P || Q) for discrete distributions
    return np.sum(np.where(q != 0, p * np.log(p / q), 0))
def main():

    """     
    p = np.array([0.8, 0.1, 0.05, 0.05]) # for a distribution, sum should be 1
    q = np.array([0.2, 0.3, 0.3, 0.2]) # 0.84
    """    
    # the following two distributions are closer to each other
    # so KL divergence will be smaller, uncomment following to test it
    p = np.array([0.8, 0.1, 0.05, 0.05])
    q = np.array([0.85, 0.05, 0.05, 0.05]) # 0.0208
    res = kl(p,q)
    print('KL divergence =', res)

if __name__ == "__main__":
    sys.exit(int(main() or 0))



=====

LayerNormTest.py

=====

import sys
import torch
import numpy as np

class NNLN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.LN = torch.nn.LayerNorm(4)
    def forward(self, x):
        out = self.LN(x)
        return out
    
def main():
    x1 = np.arange(4)
    st = np.std(x1)
    mn = np.mean(x1)
    x2 = (x1 - mn)/st
    #print(x1, ' ', st)
    print('----manual normalization')
    print(x2)
    d = torch.arange(4).float()
    print(d)
    x = d.view(1,-1)
    net = NNLN()
    z = net(x)
    print(z)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
=====

MatrixOps.py

=====

import sys
import torch

def main():
    a = torch.arange(2*2).reshape(2,2)
    print(a)
    b = torch.arange(2*2).reshape(2,2)
    print(b)
    c = a * b
    print(c) # element by element multiplication
    d = a + b # matrix addition
    print(d)
    e = torch.arange(2*3).reshape(2,3)
    print(e)
    f = torch.matmul(a,e) # matrix multiplication
    print(f)
    f1 = a @ e # also does matrix multiplication
    print(f1)
    g = torch.transpose(f,0,1)
    print(g)
    #f1 = f.reshape(1,f.shape[0], f.shape[1]) also works
    f1 = torch.unsqueeze(f,dim=0) # add a dimension in the beginning
    print(f1.shape)
    f1t = torch.transpose(f1,1,2)
    print(f1t)
    #--------batch matrix mult------------
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    res = torch.matmul(tensor1, tensor2)
    print(res.shape)
    
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

MultiNomialTest.py

=====

import sys
import torch
import torch.nn.functional as F

def main():
    logits = torch.tensor([1, 2, 3, 1, 3, 2, 3], dtype=torch.float)
    s = F.softmax(logits, dim=0)
    # to simulate top-k, lets zero out a few entries.
    s[1] = 0
    s[6] = 0
    s[1] = 0
    print(s)
    index1 = torch.multinomial(s, 1) # return probabilistically index of one top choice
    print(index1)
    index2 = torch.multinomial(s, 2) # index of top 2 choices
    print(index2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
=====

TransformerLayer.py

=====

import sys
import torch
from torch import nn
from einops import rearrange

class TransformerLayer(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.qkv = nn.Linear(d, d*3)
        self.wo = nn.Linear(d, d)
    def forward(self,x):
        x = self.qkv(x)
        q,k,v = tuple(rearrange(x,'b n (k d h)->k b h n d',k=3,h=8))
        attn = torch.einsum('b h i k, b h j k->b h i j',q,k)
        out = torch.einsum('b h i k, b h k j->b h i j',attn,v)
        out = rearrange(out,'b h n d->b n (h d)')
        out = self.wo(out)
        return out

def main():
    net = TransformerLayer(512)
    x = torch.rand((4,100,512))
    z = net(x)
    print(z.shape)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
    
=====

TriuTest.py

=====

import sys
import torch
import numpy as np
def main():
    i = 4
    j = 4
    mask = torch.ones(i, j, device = 'cuda').triu_(1).bool() # try with
    triu_(0)
    print(mask)
    print('\n')
    attn = torch.rand((4,4)).cuda()
    print(attn)
    print('\n')
    attn_masked = attn.masked_fill(mask,-np.inf)
    print(attn_masked)
if __name__ == "__main__":
    sys.exit(int(main() or 0))


"""   
File "/home/k/ub/nlp_NLP_CPEG592-X11/assignments/assignment02/code/TransformerT1/TriuTest.py", line 17, in <module>
    sys.exit(int(main() or 0))
                 ^^^^^^
File "/home/k/ub/nlp_NLP_CPEG592-X11/assignments/assignment02/code/TransformerT1/TriuTest.py", line 7, in main
    mask = torch.ones(i, j, device = 'cuda').triu_(1).bool() # try with
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/k/genkan/venv_RAG/lib/python3.11/site-packages/torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx 
"""=== THE END ===

=====

CrossEntropyTest.py

=====

import sys
import numpy as np
import torch

def main():
    a = torch.arange(10)
    a2 = a.view(-1,10)
    print(a2)
    a3 = a2[:,-3:]
    print(a3)
    # assume 3 outputs and batch size of 2, so logits = 2x3 tensor
    logits=torch.tensor([[1,3.0,5],[2,4.0,1]])
    # above indicates predicted output is 2 and 1 (index of highest value)
    print('-----logits--------')
    print(logits)
    targets=torch.tensor([2,0])
    # targets are specified as long, i.e.,
    # index of which output is to be recognized, try with [2,1] to see if loss
    decreases
    # pytorch's cross entropy loss, operates on logits (not on softmax layer)
    loss = torch.nn.functional.cross_entropy(logits,targets)
    print('\ncross entropy loss by pytorch=', loss)
    # pytorch's nll_loss (negative log likelihood loss) is similar to cross
    entropy
    # it operates on log_sofmax, rather than raw logits
    outs = torch.softmax(logits,dim=1)
    print('-----softmax------')
    print(outs)
    outs2 = torch.nn.functional.log_softmax(logits, dim=1)
    loss_nll = torch.nn.functional.nll_loss(outs2,targets)
    print('nll loss by pytorch =',loss_nll)
    # compute cross entropy ourselves
    z = (np.log(outs[0,targets[0]]) + np.log(outs[1,targets[1]]))/2
    print("\ncross entropy by our calculation=",-z)
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

EinopsTest.py

=====

import sys
import torch    
from einops import rearrange
 
def main():
    A = torch.tensor([[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]])
    B = torch.tensor([[1, 2, 1, 1],
    [3, 4, 2, 5],
    [1, 3, 6, 7],
    [1, 4, 6, 8]])
    print(A)
    print(B)
    C = torch.einsum('ij, jk -> ik', A, B) # matrix mult.
    print(C)
    C1 = torch.einsum('ij, jk -> ik', A, B) # matrix mult.
    print(C1)
    C2 = torch.einsum('ij, kj -> ik', A, B) # Ax(transpose(B) - matrix mult.
    print(C2)
    C3 = torch.einsum("ii -> i", A) # diagonal elements only
    print(C3)
    C4 = torch.einsum("ii -> ", A) # sum diagonal elements - trace
    print(C4)
    C5 = torch.einsum("ij -> j", A) # sum column elements (row wise sum)
    print(C5)
    C6 = torch.einsum('ij, ij -> ij', A, B) # element wise product
    print(C6)
    C6b = torch.einsum('ij, ij -> ', A, B) # element wise product THEN ADD ALL
    print(C6b)
    C7 = torch.einsum('ij, ij, ij -> ij', A, A, A) # cube elements
    print(C7)
    C8 = torch.einsum('ij -> ji', A)
    print(C8)
    # transpose
    C9 = torch.einsum('ij,ij -> i', A, B)
    print(C9)
    # multiply row wise and add each row
    d1 = torch.tensor([3, 5, 7, 9])
    d2 = torch.tensor([1, 2, 3, 4])
    douter = torch.einsum('i, j -> ij', d1, d2) # outer product
    print(douter)
    dinner = torch.einsum('i, i -> ', d1, d2) # inner product
    print(dinner)
    dfrobenius = torch.einsum("ij, ij -> ", A, A) # frobenius norm
    # sum of squares of all elements of a matrix
    print('Frobenius norm...')
    print(dfrobenius)
    batch_tensor_1 = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    print(batch_tensor_1)
    batch_tensor_2 = torch.arange(2 * 4 * 3).reshape(2, 3, 4)
    print(batch_tensor_2)
    dmul = torch.einsum('bij, bjk -> bik', batch_tensor_1, batch_tensor_2) #batch matrix multiplication
    print(dmul)
    dt = torch.randn((3,5,4,6,8,2,7,9)) # 8 dimensions
    print(dt.shape)
    esum = torch.einsum("ijklmnop -> p", dt)
    # marginalize or sum over dim p
    print(esum) # produces 9 numbers, try op instead of p
    kv = torch.zeros((2,1024,64)) # 2 is batch size
    q = torch.zeros((2,1024,64))
    q2 = rearrange(q,'b (n s) e->b n s e', s=16)
    print(q2.shape) #[2,64,16,64]
    q3 = rearrange(q2,'b n s e-> (b n) s e')
    print(q3.shape) #[128,16,64]
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

KLDivergenceTest.py

=====

import numpy as np
import sys

def kl(p, q):
    #Kullback-Leibler divergence D(P || Q) for discrete distributions
    return np.sum(np.where(q != 0, p * np.log(p / q), 0))
def main():

    """     
    p = np.array([0.8, 0.1, 0.05, 0.05]) # for a distribution, sum should be 1
    q = np.array([0.2, 0.3, 0.3, 0.2]) # 0.84
    """    
    # the following two distributions are closer to each other
    # so KL divergence will be smaller, uncomment following to test it
    p = np.array([0.8, 0.1, 0.05, 0.05])
    q = np.array([0.85, 0.05, 0.05, 0.05]) # 0.0208
    res = kl(p,q)
    print('KL divergence =', res)

if __name__ == "__main__":
    sys.exit(int(main() or 0))



=====

LayerNormTest.py

=====

import sys
import torch
import numpy as np

class NNLN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.LN = torch.nn.LayerNorm(4)
    def forward(self, x):
        out = self.LN(x)
        return out
    
def main():
    x1 = np.arange(4)
    st = np.std(x1)
    mn = np.mean(x1)
    x2 = (x1 - mn)/st
    #print(x1, ' ', st)
    print('----manual normalization')
    print(x2)
    d = torch.arange(4).float()
    print(d)
    x = d.view(1,-1)
    net = NNLN()
    z = net(x)
    print(z)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
=====

MatrixOps.py

=====

import sys
import torch

def main():
    a = torch.arange(2*2).reshape(2,2)
    print(a)
    b = torch.arange(2*2).reshape(2,2)
    print(b)
    c = a * b
    print(c) # element by element multiplication
    d = a + b # matrix addition
    print(d)
    e = torch.arange(2*3).reshape(2,3)
    print(e)
    f = torch.matmul(a,e) # matrix multiplication
    print(f)
    f1 = a @ e # also does matrix multiplication
    print(f1)
    g = torch.transpose(f,0,1)
    print(g)
    #f1 = f.reshape(1,f.shape[0], f.shape[1]) also works
    f1 = torch.unsqueeze(f,dim=0) # add a dimension in the beginning
    print(f1.shape)
    f1t = torch.transpose(f1,1,2)
    print(f1t)
    #--------batch matrix mult------------
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    res = torch.matmul(tensor1, tensor2)
    print(res.shape)
    
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))


=====

MultiNomialTest.py

=====

import sys
import torch
import torch.nn.functional as F

def main():
    logits = torch.tensor([1, 2, 3, 1, 3, 2, 3], dtype=torch.float)
    s = F.softmax(logits, dim=0)
    # to simulate top-k, lets zero out a few entries.
    s[1] = 0
    s[6] = 0
    s[1] = 0
    print(s)
    index1 = torch.multinomial(s, 1) # return probabilistically index of one top choice
    print(index1)
    index2 = torch.multinomial(s, 2) # index of top 2 choices
    print(index2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
=====

python-scripts-concatenated.py

=====


=====

TransformerLayer.py

=====

import sys
import torch
from torch import nn
from einops import rearrange

class TransformerLayer(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.qkv = nn.Linear(d, d*3)
        self.wo = nn.Linear(d, d)
    def forward(self,x):
        x = self.qkv(x)
        q,k,v = tuple(rearrange(x,'b n (k d h)->k b h n d',k=3,h=8))
        attn = torch.einsum('b h i k, b h j k->b h i j',q,k)
        out = torch.einsum('b h i k, b h k j->b h i j',attn,v)
        out = rearrange(out,'b h n d->b n (h d)')
        out = self.wo(out)
        return out

def main():
    net = TransformerLayer(512)
    x = torch.rand((4,100,512))
    z = net(x)
    print(z.shape)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
    
=====

TriuTest.py

=====

import sys
import torch
import numpy as np
def main():
    i = 4
    j = 4
    mask = torch.ones(i, j, device = 'cuda').triu_(1).bool() # try with
    triu_(0)
    print(mask)
    print('\n')
    attn = torch.rand((4,4)).cuda()
    print(attn)
    print('\n')
    attn_masked = attn.masked_fill(mask,-np.inf)
    print(attn_masked)
if __name__ == "__main__":
    sys.exit(int(main() or 0))


"""   
File "/home/k/ub/nlp_NLP_CPEG592-X11/assignments/assignment02/code/TransformerT1/TriuTest.py", line 17, in <module>
    sys.exit(int(main() or 0))
                 ^^^^^^
File "/home/k/ub/nlp_NLP_CPEG592-X11/assignments/assignment02/code/TransformerT1/TriuTest.py", line 7, in main
    mask = torch.ones(i, j, device = 'cuda').triu_(1).bool() # try with
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/k/genkan/venv_RAG/lib/python3.11/site-packages/torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx 
"""=== THE END ===
