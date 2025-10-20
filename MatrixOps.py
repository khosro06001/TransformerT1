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

