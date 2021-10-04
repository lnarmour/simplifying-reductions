from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':
    op = 'max'
    fp = '{[i,j]->[i]}'
    fd = '{[i,j]->[j]}'
    s = '{[i,j] : 0<=j and i-10<=j and j<=i and 2j<=i+14 }'
    #dom = BasicSet('[M,N]->{[i,j] : N=10 and M=50 }')




    # this doesn't terminate as is...

    successes = start(op, fp, s, fd, verbose=True, report_all=False)

    for success in successes:
        S = None
        for P in success.get_splits(result=set(), params=None):
            print(P)
