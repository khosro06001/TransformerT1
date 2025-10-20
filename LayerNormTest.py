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