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
"""