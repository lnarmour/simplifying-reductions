from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':


    op = 'max'
    fp = '{[i,j,k]->[i,j]}'
    fd = '{[i,j,k]->[k]}'
    s = '{ [i, j, k] : i <= 3 and k >= -4 + j and 0 <= k <= j and k <= i }'

    # this isn't generally true (need to confirm and/or fix)
    k = len(fd.split('-')[-2].split(',')) - len(fd.split('-')[-1].split(','))

    start(op, fp, s, fd, k)