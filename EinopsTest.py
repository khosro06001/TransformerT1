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

