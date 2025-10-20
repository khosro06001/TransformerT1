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


