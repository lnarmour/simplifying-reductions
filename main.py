from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':
    op = 'max'
    fp = '[N]->{[i,j,k]->[i+j]}'  # this currently take a LONG time, why?
    fd = '[N]->{[i,j,k]->[k]}'
    s = '[N]->{[i,j,k] : k<=i,j<=N+k and 0<=k<=2N}'

    successes = start(op, fp, s, fd, verbose=True, report_all=False)

    for success in successes:
        P = success.get_splits(result=set())
        [print(p) for p in P]
        print()