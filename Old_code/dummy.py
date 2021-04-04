import numpy as np

class Dummy:

    def __init__(self, E):

        self.E = E

def mama(a):
    b = a+1
    print("a=",id(a))
    print("b=", id(b))
    return a, b


if __name__ == '__main__':
    # A = np.zeros((2,2))
    #
    # E1 = Dummy(A)
    # E2 = Dummy(A)
    #
    # E1.E[1, 1] = 2
    # print(E2.E[1, 1])

    number = 4
    print("number=", id(number))
    t1, t2 = mama(number)
    print("t1=", id(t1))
    print("t2=", id(t2))
    number = 6
    print("number2=", id(number))
    print(t1, t2)