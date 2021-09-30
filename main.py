from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':
    op = 'max'
    fp = '{[i,j,k]->[i,j]}'
    fd = '{[i,j,k]->[k]}'
    s = '{[i,j,k] : k<=i,j<=10+k and 0<=k<=10}'

    #op = 'max'
    #fp = '{[i,j]->[i]}'
    #fd = '{[i,j]->[j]}'
    #s = '{[i,j] : 0<=j and i-10<=j and j<=i and 2j<=i+14 }'

    successes = start(op, fp, s, fd, verbose=False, report_all=False)

    for success in successes:
        P = success.get_splits()
        [print(p) for p in P]
        print()