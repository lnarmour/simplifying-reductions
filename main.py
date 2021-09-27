from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':

    op = 'max'
    fp = '[N,M]->{[i,j,k]->[i,j]}'
    fd = '[N,M]->{[i,j,k]->[k]}'
    s = '[N,M]->{[i,j,k] : k<=i,j<=N+k and 0<=k<=M}'

    op = 'max'
    fp = '[N]->{[i,j]->[i]}'
    fd = '[N]->{[i,j]->[j]}'
    s = '[N]->{[i,j] : 0<=j and i-N<=j and j<=i and j<=3N/2 }'

    # this isn't generally true (need to confirm and/or fix)
    k = len(fd.split('-')[-2].split(',')) - len(fd.split('-')[-1].split(','))

    successes = start(op, fp, s, fd, k)

    for s in successes:
        splits = s.get_splits()
        x = 0