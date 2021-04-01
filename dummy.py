import numpy as np

class Dummy:

    def __init__(self, E):

        self.E = E


if __name__ == '__main__':
    A = np.zeros((2,2))

    E1 = Dummy(A)
    E2 = Dummy(A)

    E1.E[1, 1] = 2
    print(E2.E[1, 1])