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