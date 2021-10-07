from simplify.simplify import start
from simplify.face_lattice import *
from simplify.homothety import *


if __name__ == '__main__':
    op = 'max'
    fp = '[N]->{[i,j,k]->[j]}'
    fd = '[N]->{[i,j,k]->[k]}'
    s = '[N]->{[i,j,k] : k<=i,j<=N+k and 0<=k<=5N}'

    # visulaization
    dom = BasicSet('[N]->{[i,j,k] : N=5}')




    # this doesn't terminate as is...

    successes = start(op, fp, s, fd, verbose=True, report_all=False)

    for success in successes:
        S = None
        for P in success.get_splits(result=set(), params=dom):
            print(P)
