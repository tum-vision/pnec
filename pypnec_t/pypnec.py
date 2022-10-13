from distutils.command import build

import numpy as np

from ../build/python import pypnec


def main():
    print("Hello World")
    print(dir(pypnec))
    print(pypnec.add(1, 2))
    a = np.random.rand(3, 3)
    print(a)
    print(pypnec.mat2(a))
    print(pypnec.matrices([a]))


if __name__ == "__main__":
    main()
