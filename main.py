from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':
    op = 'max'
    fp = '[M,N]->{[i,j]->[i]}'
    fd = '[M,N]->{[i,j]->[j]}'
    s = '[M,N]->{[i,j] : 0<=j and i-N<=j and j<=i and j<=M and N<M }'

    successes = start(op, fp, s, fd, verbose=True, report_all=False)

    for success in successes:
        P = success.get_splits(result=set())
        [print(p) for p in P]
        print()