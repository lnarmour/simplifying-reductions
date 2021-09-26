from . face_lattice import *
from . homothety import *
from islpy import *
from copy import deepcopy
from enum import Enum
from itertools import combinations
import numpy as np
import sys
import argparse

global CNT
global RANK
CNT = 0


def print_successes(success, indent=0):
    if success.action == Action.NONE:
        print('{}{}'.format(indent * ' ', success))
        return

    print('{}{}'.format(indent * ' ', success))
    for split in success.children:
        for child_success in split:
            print_successes(child_success, indent + 2)


def inverse(op):
    store = {
        '+': '-',
        'max': None
    }
    if op not in store:
        raise Exception('Operator "{}" not supported yet.'.format(op))
    return store[op]


def count_vars_of_type(isl_set_map, match_type=None):
    return len([i for i in isl_set_map.get_var_dict().items() if i[1][0] == match_type])


# TODO can remove this?
# creates the matrix representation of f
# if f = '{[i,j,k]->[i]}', then this returns [[1,0,0]]
# if f = '{[i,j,k]->[i+k,j]}', then this returns [[1,0,1],[0,1,0]]
def map_to_matrix(f):
    mat = []
    for c in f.get_constraints():
        vars = c.get_var_dict()
        index_vals = [-1 * int(c.get_coefficient_val(v[0], v[1]).to_str()) for k, v in vars.items() if
                      v[0] != dim_type.param]
        mat.append(index_vals)
    return np.array(mat)


def mat_to_set(mat, lattice, condition='='):
    indices = ','.join([str(i[0]) for i in lattice.indices])
    params = ','.join([str(i[0]) for i in lattice.params])
    constraints = []
    for c in mat:
        constraints.append('+'.join(['{}{}'.format(a[0],a[1]) for a in zip(c, indices.split(','))]) + '{}0'.format(condition))
    s = '{{[{}] : {}}}'.format(indices, ' and '.join(constraints))
    s = '{}{}'.format('[{}]->'.format(params) if params else '', s)
    return BasicSet(s)


# convert base-10 number n to base-d
def d_ary(d, n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, d)
        nums.append(str(r))
    return ''.join(reversed(nums))


def enumerate_labels(facets, labels):
    LABEL = {i:labels[i] for i in range(len(labels))}
    num_labels = len(labels)
    num_facets = len(facets)
    for i in range(num_labels ** num_facets):
        labels = [LABEL[int(d_ary(num_labels, i).zfill(num_facets)[j])] for j in range(num_facets)]
        yield labels


def rho_from_labels(faces, parent, Lp, lattice, labels, fd):
    # check if this combo is possible given the feasible_rho
    # c1     c2     c3
    # 'ADD', 'ADD', 'INV'
    # c1*rho>=0 and c2*rho>=0 and c3*rho=0  and in ker(fd) and it saturates current node
    # {[i,j,k] : j>=0 and i-j-k>=0 and k=0  and i=0   }  <- is this empty?  if yes, then prune out this combo
    # if no, then check that it's "in ker(fd)
    C = lattice.C
    map = {'ADD': '>', 'INV': '=', 'SUB': '<'}

    bset = Lp
    # must be in ker(fd)
    bset = bset.intersect(mat_to_set(map_to_matrix(fd), lattice))

    if bset.is_empty():
        raise Exception('Lp * ker(fd) should not be empty')

    satsified_faces = []
    for face, label in zip(faces, labels):
        # select constraints representing this face, and create the set according to its label
        next_bset = mat_to_set(C[np.array(list(face-parent)),:lattice.num_indices], lattice, map[label])
        bset = bset.intersect(next_bset)

        if (bset - lattice.origin).is_empty():
            break
        else:
            satsified_faces.append(face)

    if len(satsified_faces) == len(faces):
        feasible_rho = bset - lattice.origin
        return feasible_rho.sample_point(), satsified_faces, None

    #
    return None, satsified_faces, face



def prune_combos(label_combos):
    combos = [[1 if l == 'ADD' else 0 for l in c[:-1]] for c in label_combos]
    combos.sort(key=lambda c: np.sum(c), reverse=1)
    # sort by combos with most ADD faces first
    # discard a combo if its ADD faces can
    ret = []
    for i,combo in enumerate(combos):
        _combo = np.array(combo)
        for other in combos[i+1:]:
            other = np.array(other)
            _combo = _combo - other
        ret.append(combo)
    # TODO - return only label_combos rows that match ret
    unique_combos = []
    for lc in label_combos:
        labels = [1 if l == 'ADD' else 0 for l in lc[:-1]]
        if labels in ret:
            unique_combos.append(lc)
    return unique_combos


def add_ineq_constraint(bset, aff):
    constraint = Constraint.inequality_from_aff(aff)
    cut_bset = bset.add_constraint(constraint).remove_redundancies()
    return cut_bset


def simplify(k=None, fp_str=None, fd_str=None, node=None, lattice=None, legal_labels=None, iss_parent_nodes=[], do_decomp=True, do_splitting=True):
    global CNT
    def pprint(*args, **kwargs):
        if len(iss_parent_nodes) == 0:
            node_label = '@{} '.format(set(node) if node else '{}')
        else:
            node_label = '@{}:@{} '.format(set(iss_parent_nodes[-1][0]) if iss_parent_nodes[-1][0] else '{}',
                                          set(node) if node else '{}')
        print(node_label, end='')
        print(*args, **kwargs)

    successful_combos = []

    if 0:
        pprint('STEP A.0 - have we seen this problem before?')
        pprint()
        for parent_node in reversed(iss_parent_nodes):
            parent_facet, parent_bset = parent_node
            current_bset = lattice.create_facet_bset(node)
            if bsets_are_homothetic(parent_bset, current_bset):
                pprint('SUCCESS')
                bsets_are_homothetic(parent_bset, current_bset)
                return [Success(Action.HOMOTHETY, node, fp_str, iss_parent_nodes=iss_parent_nodes.copy(), notes='at {}'.format(current_bset))]
        pprint('No homothety detected.')
        pprint()

    pprint('STEP A.1 - if k==0 return else continue. k={}'.format(k))
    pprint()
    if k == 0:
        pprint('SUCCESS')
        return [Success(Action.NONE, node, fp_str, iss_parent_nodes=iss_parent_nodes.copy())]

    fp = BasicMap(fp_str)
    fd = BasicMap(fd_str)
    pprint('node = {}'.format(set(node) if node else '{}'))
    pprint('fp = {}'.format(fp_str))
    pprint('fd = {}'.format(fd_str))
    pprint()

    pprint('STEP A.2 - identify boundaries given fp')
    pprint()
    Lp = lattice.get_Lp(node)  # theorem 2
    for facet in lattice.get_facets(node):
        pprint(set(facet), lattice.boundary_label(facet, fp, Lp))
    pprint()

    pprint('STEP A.3 - construct list of candidate facets (i.e., non-boundary facets)')
    pprint()
    candidate_facets = lattice.get_candidate_facets(node, fp)
    pprint('candidate_facets = {}'.format([set(cf) for cf in candidate_facets]))
    pprint()

    pprint('STEP A.4 - determine all possible combos')
    pprint()
    label_combos = list()
    header = ['{}'.format(set(f)) for f in candidate_facets]
    pprint(header)
    pprint('-' * len(str(header)))
    for labels in enumerate_labels(candidate_facets, legal_labels):
        rho, satisfied_facets, problem_facet = rho_from_labels(candidate_facets, node, Lp, lattice, labels, fd)
        if rho and not rho.is_void():
            label_combos.append(labels + [rho])
            pprint('{}  possible -> rho = {}'.format(labels, rho))
        else:
            pprint('{}  impossible'.format(labels))
    pprint()

    if label_combos:
        pprint('STEP A.5 - prune out redundant possible combos')
        pprint()
        pprint(header)
        pprint('-' * len(str(header)))
        unique_label_combos = prune_combos(label_combos)
        for combo in unique_label_combos:
            labels, rho = combo[:-1], combo[-1]
            pprint('{}  -> rho = {}'.format(labels, rho))
        pprint()

        pprint('STEP A.6 - incorporate boundary facets')
        pprint()
        boundary_facets = [f for f in lattice.get_facets(node) if f not in candidate_facets]
        header = '{} {}'.format(header, [str(set(bf)) for bf in boundary_facets])
        pprint(header)
        pprint('-' * len(str(header)))
        full_label_combos = []
        for combo in unique_label_combos:
            labels, rho = combo[:-1], combo[-1]
            boundary_labels = []
            full_label_combo = deepcopy(labels)
            for boundary_facet in boundary_facets:
                bl = lattice.label(boundary_facet, node, rho)
                boundary_labels.append(bl)
                full_label_combo.append(bl)
            pprint('{} {}  -> rho = {}'.format(labels, boundary_labels, rho))
            full_label_combo.append(rho)
            full_label_combos.append(full_label_combo)
        pprint()

        pprint('STEP A.7 - recurse into "ADD" and "inward" boundary facets')
        pprint()
        for combo in full_label_combos:
            labels,rho = combo[:-1],combo[-1]
            abort = None
            rets = []
            for label,facet in zip(labels,candidate_facets + boundary_facets):
                if label != 'ADD' and label != 'inward':
                    continue
                pprint('recursing into {} facet'.format(set(facet)))
                pprint()
                ret = simplify(k=k-1, fp_str=fp_str, fd_str=fd_str, node=facet, lattice=lattice, legal_labels=legal_labels,
                               iss_parent_nodes=iss_parent_nodes.copy(),
                               do_decomp=do_decomp,
                               do_splitting=do_splitting)
                abort = not ret
                if abort:
                    break
                rets.append(ret)
            if not abort:
                successful_combos.append(Success(Action.RHO, node, fp_str, candidate_facets + boundary_facets, combo, children=rets, iss_parent_nodes=iss_parent_nodes.copy()))
            pprint()
        pprint()

    pprint('STEP C - reduction decomposition')
    pprint()

    pprint('STEP C.1 - is decomposition feasible?')
    pprint()
    lhs_num_vars = count_vars_of_type(fp, dim_type.in_)
    rhs_num_vars = fp.n_constraint()
    decomp_feasible = lhs_num_vars - rhs_num_vars > 1
    if decomp_feasible:
        pprint('STEP C.2 - enumerate WEAK boundaries')
        pprint()
        weak_boundaries = lattice.get_weak_boundary_facets(node, fp)
        for weak_boundary in weak_boundaries:
            pprint(set(weak_boundary), BoundaryLabel.WEAK_BOUNDARY)
        pprint()

        # given a weak boundary (i.e., a vector in the intersection)
        decomps = set()
        for weak_facet in weak_boundaries:
            pprint('STEP C.3 - decompose fp based to adjust facet {}'.format(set(weak_facet)))
            pprint()
            fp1_str = lattice.decompose_projection_from_weak_facet(weak_facet, fp, Lp)
            if not fp1_str:
                pprint("decomposition not feasible for fp = {} and weak_facet = {}".format(fp_str, set(weak_facet)))
                pprint()
                continue
            if fp1_str in decomps:
                pprint('already processed decomposition {} before, skipping'.format(fp1_str))
                continue
            decomps.add(fp1_str)
            pprint('fp = fp2 * fp1')
            pprint('fp1 = {}'.format(fp1_str))
            pprint()
            ret = simplify(k=k, fp_str=fp1_str, fd_str=fd_str, node=node, lattice=lattice,
                           legal_labels=legal_labels,
                           iss_parent_nodes=iss_parent_nodes.copy(),
                           do_decomp=do_decomp,
                           do_splitting=do_splitting)
            #
            # TODO - implement residual reduction with fp2
            #
            if ret:
                successful_combos.append(
                    Success(Action.DECOMPOSITION, node, fp_str, fp_decomp=fp1_str, children=[ret],
                            iss_parent_nodes=iss_parent_nodes.copy()))
                break
    else:
        pprint("decomposition not feasible for fp = {}".format(fp_str))
    pprint()

    if len(successful_combos) > 0:
        return successful_combos

    pprint('STEP D - index set splitting')
    pprint()

    node_bset = lattice.create_facet_bset(node)
    current_rank = lattice.compute_rank(lattice.bset)

    for f in [fp, fd]:
        ker_f = lattice.ker(f)
        ker_f_rank = lattice.compute_rank(ker_f)

        # in 3D, want to make 2D planes
        # to make 2D planes, either:
        # - we already have enough from ker_f (i.e., rank(ker_f) = current_rank - 1)
        #   in which case we just need to translate to a vertex
        # - we need additional vertices - to make 2D plane, if rank(ker_f) = 1
        #   then we need 1 more vertex
        split_affs = []

        if ker_f_rank == RANK - 1:
            for vertex in lattice.get_vertices_on(node):
                # translate ker_f to vertex and take its equality constraint as aff
                trunc_vertex = lattice.truncate_vertex(vertex)
                m = build_map(ker_f.get_space(), trunc_vertex, None, lattice.num_params, lattice.num_indices)
                ker_f_at_v0 = ker_f.apply(m)
                equalities = [c for c in ker_f_at_v0.get_constraints() if c.is_equality()]
                assert len(equalities) == 1
                split_aff = equalities[0].get_aff()
                split_affs.append(split_aff)
        else:
            num_vertices_needed = RANK - lattice.compute_rank(ker_f) - 1
            for vertices in combinations(lattice.get_vertices_on(node), num_vertices_needed):
                split_aff = lattice.make_hyperplane(RANK - 1, ker_f, vertices)
                if not split_aff:
                    continue
                split_affs.append(split_aff)

        for split_aff in split_affs:
            r_aff = split_aff
            l_aff = r_aff.neg()
            l_aff = l_aff.set_constant_val(l_aff.get_constant_val() - 1)

            l_cut_bset = add_ineq_constraint(node_bset, l_aff)
            r_cut_bset = add_ineq_constraint(node_bset, r_aff)
            l_cut_rank = lattice.compute_rank(l_cut_bset)
            r_cut_rank = lattice.compute_rank(r_cut_bset)
            either_is_empty = l_cut_bset.is_empty() or r_cut_bset.is_empty()
            either_has_different_rank = l_cut_rank != current_rank or r_cut_rank != current_rank
            if either_is_empty or either_has_different_rank:
                pprint('cut "{}" is invalid'.format(l_aff))
                continue

            pprint('cut "{}" is valid'.format(l_aff))
            pprint('node = BasicSet("{}")'.format(node_bset))
            pprint('L = BasicSet("{}")'.format(l_cut_bset))
            pprint('R = BasicSet("{}")'.format(r_cut_bset))

            l_lattice = FaceLattice(bset=l_cut_bset)
            r_lattice = FaceLattice(bset=r_cut_bset)

            pprint('left side lattice')
            l_lattice.pretty_print_constraints()
            l_lattice.pretty_print()
            pprint()
            pprint('right side lattice')
            r_lattice.pretty_print_constraints()
            r_lattice.pretty_print()
            pprint()

            # recurse into each side of the cut, they both must return success for this
            # cut to be valid
            if not iss_parent_nodes:
                iss_parent_nodes = []
            iss_parent_nodes.append((node, lattice.create_facet_bset(node)))
            l_ret = simplify(k=k, fp_str=fp_str, fd_str=fd_str, node=l_lattice.root[0], lattice=l_lattice,
                             legal_labels=legal_labels,
                             iss_parent_nodes=iss_parent_nodes.copy(),
                             do_decomp=do_decomp,
                             do_splitting=do_splitting)

            r_ret = simplify(k=k, fp_str=fp_str, fd_str=fd_str, node=r_lattice.root[0], lattice=r_lattice,
                             legal_labels=legal_labels,
                             iss_parent_nodes=iss_parent_nodes.copy(),
                             do_decomp=do_decomp,
                             do_splitting=do_splitting)

            if l_ret and r_ret:
                successful_combos.append(
                    Success(Action.INDEX_SET_SPLIT, node, fp_str, iss_parent_nodes=iss_parent_nodes.copy(),
                            children=[l_ret, r_ret],
                            notes='at {}'.format(split_aff),
                            l_cut_bset=l_cut_bset,
                            r_cut_bset=r_cut_bset))
                break
        if successful_combos:
            break

    return successful_combos


class Success:

    def __init__(self, action, node, fp_str, facets=None, labelings=None, iss_parent_nodes=None,
                 splits=None, children=list(), fp_decomp=None,
                 notes=None, l_cut_bset=None, r_cut_bset=None):
        self.action = action
        self.node = node
        self.fp_str = fp_str
        self.facets = facets
        self.labelings = labelings
        self.iss_parent_nodes = iss_parent_nodes
        self.children = children
        self.fp_decomp = fp_decomp
        self.notes = notes
        self.l_cut_bset = l_cut_bset
        self.r_cut_bset = r_cut_bset

    def __str__(self):
        if not self.iss_parent_nodes:
            ret = '@{}, {}'.format(set(self.node) if self.node else '{}', self.fp_str)
        else:
            ret = '@{}:@{}, {}'.format(set(self.iss_parent_nodes[-1][0]) if self.iss_parent_nodes[-1][0] else '{}',
                                       set(self.node) if self.node else '{}', self.fp_str)
        ret = '{} :: {}'.format(ret, self.action)
        if self.action == Action.DECOMPOSITION:
            ret = '{}, {}'.format(ret, self.fp_decomp)
        if self.facets and self.labelings:
            rho = self.labelings[-1]
            ret = '{}, {}, {}, {}'.format(ret, rho, [set(f) for f in self.facets], self.labelings[:-1])
        ret = '{} {}'.format(ret, self.notes if self.notes else '')
        return ret

    def get_splits(self):
        if self.action == Action.INDEX_SET_SPLIT:
            yield self.l_cut_bset, self.r_cut_bset
        for split in self.children:
            for child_success in split:
                return child_success.get_splits()


class Action(Enum):
    INDEX_SET_SPLIT = 1
    RHO = 2
    DECOMPOSITION = 3
    HOMOTHETY = 4
    NONE = 5


def start(op, fp, s, fd, k):
    legal_labels = ['ADD', 'INV']
    if inverse(op):
        legal_labels.append('SUB')

    lattice = FaceLattice(s)

    global RANK
    RANK = lattice.compute_rank(lattice.bset)

    lattice.chamber = 0
    print('constraints:')
    lattice.pretty_print_constraints()
    print('---')
    ret = simplify(k=k, fp_str=fp, fd_str=fd, node=lattice.root[0], lattice=lattice, legal_labels=legal_labels, do_decomp=False, do_splitting=True)
    print('---')
    print('op:', op)
    print('fp:', fp)
    print('fd:', fd)
    print('s: ', s)
    print('---')
    print('constraints:')
    lattice.pretty_print_constraints()
    print('---')
    print('Simplifications')
    for success in ret:
        print_successes(success)
    x = 0
    return ret


if __name__ == '__main__':
    homothety_tests()

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='compute face lattice only')
    parser.add_argument('-d', action='store_true', help='debug')
    parser.add_argument('-s', help='input set')
    parser.add_argument('-op', help='reduction operation', default='max')
    parser.add_argument('-fp', help='projection function', default='{[i,j,k]->[i]}')
    parser.add_argument('-fd', help='dependence function', default='{[i,j,k]->[k]}')

    args = parser.parse_args()

    if args.l:
        show_face_lattice(args.set)
        sys.exit(0)

    op = args.op
    fp = args.fp
    s = args.s
    fd = args.fd

    if 1:
        # homothetic termination
        op = 'max'
        fp = '{[i,j]->[i]}'
        s = '[N]->{[i,j] : 0<=i,j and 3i-3N<=j<=i and 0<N }'
        fd = '{[i,j]->[j]}'
    if 1:
        # 3D parallelepiped
        op = 'max'
        fp = '[N,M]->{[i,j,k]->[i]}'
        s = '[N,M]->{[i,j,k] : k<=i,j<=4+k and 0<=k<=8 }'
        fd = '[N,M]->{[i,j,k]->[k]}'
    if 1:
        op = 'max'
        fp = '[N]->{[i,j]->[i]}'
        fd = '[N]->{[i,j]->[j]}'
        s = '[N] -> {[i,j] : 0<=j and i-N<=j and j<=i and j<=3N/2 }'
    if 1:
        op = 'max'
        fp = '{[i,j]->[i]}'
        fd = '{[i,j]->[j]}'
        s = '{[i,j] : 0<=j and i-4<=j and j<=i and j<=6 }'
    if 1:
        op = 'max'
        fp = '{[i,j,k]->[i]}'
        fd = '{[i,j,k]->[k]}'
        s = '{ [i, j, k] : i <= 3 and k >= -4 + j and 0 <= k <= j and k <= i }'

    # this isn't generally true (need to confirm and/or fix)
    k = len(fd.split('-')[-2].split(',')) - len(fd.split('-')[-1].split(','))

    start(op, fp, s, fd, k)