from islpy import *
from enum import Enum
from itertools import combinations
import networkx as nx
from . homothety import *


class BoundaryLabel(Enum):
    STRICT_BOUNDARY = 1
    WEAK_BOUNDARY = 2
    NON_BOUNDARY = 3


class CustomMultiAff:

    def __init__(self, multi_aff, idx, num_indices):
        self.multi_aff = multi_aff
        self.idx = idx
        self.affs = tuple([multi_aff.get_aff(i) for i in range(num_indices)])

    def __hash__(self):
        return hash(self.affs)

    def __eq__(self, other):
        return self.affs == other.affs


def face_lattice_examples():
    S = [
        # '{ [i,j,k] : 0<=i,j,k and k+j>=i }',
        '[N] -> { [i,j] : N>=5 and 0<=i,j and i+j<=N }',
        '[N,M] -> { [i,j] : N>=5 and 0<=i,j and i+j<=N and j<=M }',
        '[N] -> { [i,j,k] : 0<=i,j and k=0 and i+j<N }',
        '[N] -> { [i,j,k] : 0<=i,j and k=i-15 and i+j<N }'
    ]
    for i, s in enumerate(S):
        show_face_lattice(s)


def show_face_lattice(s, i=0):
    fl = FaceLattice(s)
    print('input:')
    print(fl.s)
    print()
    print('constraints:')
    fl.pretty_print_constraints()
    for j, chamber in enumerate(fl.chambers):
        if fl.params:
            print('chamber-{}: {}'.format(j, chamber.get_domain()))
        fl.pretty_print()


# face lattice
class FaceLattice:

    def __init__(self, s=None, bset=None):
        if not s and not bset:
            return
        if s:
            self.s = s
        self.bset = bset.remove_redundancies() if bset else BasicSet(s).remove_redundancies()
        assert self.bset.is_bounded()
        self.space = self.bset.get_space()
        self.isl_C = list(self.bset.get_constraints())

        # construct constraints matrix
        self.C_type, self.C = self.get_constraints(self.bset)
        self.num_constraints = len(self.C)

        # compute vertex lattice nodes from ISL vertices
        self.chamber = 0
        self.chambers = []
        self.bset.polyhedral_hull().compute_vertices().foreach_cell(self.chambers.append)
        space = self.bset.get_space()
        self.space = space
        self.indices = [(k, v) for k, v in space.get_id_dict().items() if v[0] == dim_type.out]
        self.num_indices = len(self.indices)
        self.params = [(k, v) for k, v in space.get_id_dict().items() if v[0] == dim_type.param]
        self.num_params = len(self.params)
        eye = list([list([int(v) for v in row]) for row in np.eye(len(self.params), dtype=int)])

        self.vertex_nodes = {}
        for i, chamber in enumerate(self.chambers):
            vertices = []
            self.chambers[i].foreach_vertex(vertices.append)
            nodes = []
            for v in vertices:
                node = set()
                for k,c in enumerate(self.isl_C):
                    if c.is_equality():
                        continue
                    if self.vertex_saturates_constraint(v, c):
                        node.add(k)
                nodes.append(node)
            self.vertex_nodes[i] = (chamber, vertices, nodes)

        self.dim_P = self.compute_rank(self.bset)
        self.kfaces = {}
        self.graph = {}
        self.root = {}
        self.explored = {}
        # There is actually one face lattice per chamber
        for i in self.vertex_nodes:
            self.face_lattice(chamber=i)
        self.origin = self.zero()
        self.basis_sets = self.construct_basis_vectors()

    def get_vertices(self, facet):
        maffs = [v.get_expr() for v in self.vertex_nodes[self.chamber][1]]
        vertex_facets = self.vertex_nodes[self.chamber][2]
        maffs = [maff for maff,s in zip(maffs, vertex_facets) if facet.issubset(s)]
        maffs.sort(key=lambda v: multi_aff_to_vec(v, self.num_indices, self.num_params))
        return maffs

    def get_chamber_domain(self):
        return self.chambers[self.chamber].get_domain()

    def get_chamber_domain_constraints(self):
        C = []
        empty_aff = Constraint.equality_alloc(self.space).get_aff()

        for c in self.get_chamber_domain().get_constraints():
            aff = c.get_aff()
            new_aff = empty_aff
            for i in range(self.num_params):
                new_aff = new_aff.set_coefficient_val(dim_type.param, i, aff.get_coefficient_val(dim_type.param, i))
            new_aff = new_aff.set_constant_val(aff.get_constant_val())

            if c.is_equality():
                C.append(Constraint.equality_from_aff(new_aff))
            else:
                C.append(Constraint.inequality_from_aff(new_aff))

        return C

    def zero(self):
        o = BasicSet.universe(self.space)
        c = Constraint.equality_alloc(self.space)
        for i in range(self.num_indices):
            o = o.add_constraint(c.set_coefficient_val(dim_type.out, i, 1))
        return o

    def construct_basis_vectors(self):
        ret = []
        # construct sets spanned by basis vectors in d-dimensional polyhedron
        eq = Constraint.equality_alloc(self.space)
        for i in range(self.num_indices):
            bset = BasicSet.universe(self.space)
            for j in range(self.num_indices):
                val = 0 if i == j else 1
                bset = bset.add_constraint(eq.set_coefficient_val(dim_type.out, j, val))
            ret.append(bset)
        return ret

    def build_map(self, vertex):
        space = Space.alloc(self.space.get_ctx(), self.num_params, self.num_indices, self.num_indices)
        for name, tuple in self.space.get_var_dict().items():
            type, pos = tuple
            space = space.set_dim_name(type, pos, name)
            if type == dim_type.out:
                space = space.set_dim_name(dim_type.in_, pos, name)

        m = BasicMap.universe(space)

        for i in range(self.num_indices):
            aff = vertex.get_aff(i)
            # can only set integer values so scale all dims by lcm denominators
            max_den = aff.get_constant_val().get_den_val().to_python()
            if self.num_params > 0:
                max_den = np.lcm.reduce([max_den] + [aff.get_coefficient_val(dim_type.param, j).get_den_val().to_python() for j in range(self.num_params)])
            constraint = Constraint.alloc_equality(space)
            constraint = constraint.set_coefficient_val(dim_type.out, i, max_den * -1)
            constraint = constraint.set_coefficient_val(dim_type.in_, i, max_den)
            for j in range(self.num_params):
                val = aff.get_coefficient_val(dim_type.param, j)
                constraint = constraint.set_coefficient_val(dim_type.param, j, val * max_den)
            constraint = constraint.set_constant_val(aff.get_constant_val() * max_den)
            m = m.add_constraint(constraint)

        if len(m.get_constraints()) == self.num_indices:
            return m
        else:
            return None


    def make_hyperplane(self, d, vertex, f):
        # compute d-dimensional hyperplane in null space of "f" passing through "vertex"
        ker_f = self.ker(f)
        k = self.compute_rank(ker_f)

        hyperplanes = []
        if d - k > 0:
            for C in combinations(self.basis_sets, d - k):
                hyperplane = ker_f
                for basis_set in C:
                    hyperplane = hyperplane.union(basis_set).polyhedral_hull()
                if self.compute_rank(hyperplane) < d:
                    continue
                hyperplanes.append(hyperplane)
        else:
            hyperplanes.append(ker_f)

        m = self.build_map(vertex)
        if not m:
            return list()

        # translate each hyperplane to vertex
        hyperplanes = [h.apply(m).polyhedral_hull() for h in hyperplanes]

        affs_set = set()
        affs = []
        for h in hyperplanes:
            equality_constraints = [c for c in h.get_constraints() if c.is_equality()]
            assert len(equality_constraints) == 1
            equality = equality_constraints[0]
            aff = equality.get_aff()
            if aff not in affs_set:
                affs.append(aff)
                affs_set.add(aff)

        return affs



    def vertex_saturates_constraint(self, vertex, constraint):
        # returns true if vertex saturate the k'th constraint
        v = self.vertex_to_vec(vertex)
        c = self.constraint_to_vec(constraint)
        s = np.matmul(v, c.transpose())
        return np.all(s == 0)

    def constraint_to_vec(self, constraint):
        indices = [constraint.get_coefficient_val(dim_type.out, c) for c in range(self.num_indices)]
        params = [constraint.get_coefficient_val(dim_type.param, c) for c in range(self.num_params)]
        constant = [constraint.get_constant_val()]
        return np.array(indices + params + constant)

    def vertex_to_vec(self, vertex):
        # space: [N,M] -> { [i,j,k,l] : ... }
        #   i j k l
        # N . . . . 1 0 0
        # M . . . . 0 1 0
        # d . . . . 0 0 1
        #
        maff = vertex.get_expr()
        vec = np.zeros((self.num_params + 1, self.num_indices), dtype=Val)
        for i in range(self.num_indices):
            aff = maff.get_aff(i)
            for j in range(self.num_params):
                vec[j, i] = aff.get_coefficient_val(dim_type.param, j)
            vec[-1, i] = aff.get_constant_val()
        eye = np.zeros((self.num_params + 1, self.num_params + 1), dtype=Val)
        for i in range(self.num_params + 1):
            for j in range(self.num_params + 1):
                eye[i, j] = Val(1) if i == j else Val(0)
        return np.hstack((vec, eye))

    def find_kfaces(self, chamber, facet):
        current_rank = self.compute_rank(self.create_facet_bset(facet))
        next_rank = current_rank
        for i in range(1, len(facet)):
            for c in combinations(facet, i):
                # unsaturate constraints in c
                candidate_facet = frozenset([i for i in facet if i not in c])
                if candidate_facet in self.considered:
                    continue
                self.considered.add(candidate_facet)
                bset = self.create_facet_bset(candidate_facet)
                rank = self.compute_rank(bset)
                if rank > current_rank:
                    next_rank = rank
                    candidate_facet = [c for c in candidate_facet]
                    if rank not in self.kfaces[chamber]:
                        self.compute_rank(bset)
                    self.kfaces[chamber][rank].append(candidate_facet)
            current_rank = max(current_rank, next_rank)

    def face_lattice(self, chamber=0):
        nodes = [frozenset(node) for node in self.vertex_nodes[chamber][2]]

        # find all k-faces of P starting from vertex nodes
        # think of a k-face as a set of dimension (rank) k
        self.kfaces[chamber] = {0: nodes}
        for rank in range(1, self.dim_P + 1):
            self.kfaces[chamber][rank] = []
        self.considered = set()
        for node in nodes:
            self.find_kfaces(chamber, node)
        self.kfaces[chamber][self.dim_P] = [frozenset()]
        self.build_graph(chamber)

    def build_graph(self, chamber=0):
        # construct lattice data structure from result
        self.graph[chamber] = nx.DiGraph().copy()
        self.explored[chamber] = dict()

        self.kfaces[chamber] = {k:v for k, v in self.kfaces[chamber].items() if v}
        Ks = list(reversed([k for k in self.kfaces[chamber]]))
        self.root[chamber] = frozenset(self.kfaces[chamber][Ks[0]][0])
        self.graph[chamber].add_node(self.root[chamber])
        for i,k in enumerate(Ks[1:]):
            prev_k = Ks[i]
            for s in self.kfaces[chamber][k]:
                node = frozenset(s)
                self.graph[chamber].add_node(node)
                if i == 0:
                    self.graph[chamber].add_edge(self.root[chamber], node)
                else:
                    for prev_s in self.kfaces[chamber][prev_k]:
                        prev_node = frozenset(prev_s)
                        if prev_node.issubset(node):
                            self.graph[chamber].add_edge(prev_node, node)

    def create_facet_bset(self, facet):
        # facet = constraints to be saturated
        fset = self.bset
        for c in facet:
            saturated_c = Constraint.equality_from_aff(self.isl_C[c].get_aff())
            fset = fset.add_constraint(saturated_c)
        fset.remove_redundancies()
        return fset

    def pretty_print_constraints(self):
        ret = '\n'.join([str(c) for c in self.isl_C])
        cnt = 0
        for row, constraint in zip(str(self.C).split('\n'), ret.split('\n')):
            print('c{}\t{}\t{}'.format(cnt, row, constraint))
            cnt += 1
        print()

    def compute_rank(self, bset):
        bset = bset.remove_redundancies()
        num_indices = len([v for v in bset.get_space().get_var_dict().items() if v[1][0] == dim_type.out])
        num_equalities = len([c for c in bset.get_constraints() if c.is_equality()])
        return num_indices - num_equalities

    def get_constraints(self, bset):
        C = []
        isl_C = []
        for c in bset.get_constraints():
            isl_C.append(c)
            vars = c.get_var_dict()
            index_vals = [int(c.get_coefficient_val(v[0], v[1]).to_str()) for k, v in vars.items() if
                          v[0] != dim_type.param]
            param_vals = [int(c.get_coefficient_val(v[0], v[1]).to_str()) for k, v in vars.items() if
                          v[0] == dim_type.param]
            const_val = [int(c.get_constant_val().to_str())]
            polylib_prefix = [0] if c.is_equality() else [1]
            C.append(polylib_prefix + index_vals + param_vals + const_val)
        C.append([0] * (1 + len(index_vals) + len(param_vals)) + [1])
        C = np.array(C)
        # first column of C indicates whether constraint is equality or inequality
        # last column is the constant
        C_type = np.array(C[:-1, 0])
        C = np.array(C[:-1, 1:])
        return C_type, C

    def get_num_params_indices(self, bset):
        return len(self.params), len(self.indices)

    def get_Lp(self, node=None):
        # start with universe
        Lp = BasicSet.universe(self.bset.get_space())
        # add equalities present in root
        equalities = list(np.where(self.C_type == 0)[0])

        # add equalities saturated in facet
        if node:
            equalities += list(node)

        # TODO - I'm sure there is a better way to do this below (w/ from Aff)

        indices = ','.join([str(i[0]) for i in self.indices])
        params = ','.join([str(i[0]) for i in self.params])
        for c in list(equalities):
            mat = self.C[[c], :self.num_indices]
            constraints = []
            for c in mat[:len(indices)]:
                constraints.append('+'.join(['{}{}'.format(a[0], a[1]) for a in zip(c, indices.split(',')+params.split(','))]) + '=0')
            s = '{{[{}] : {}}}'.format(indices, ' and '.join(constraints))
            s = '{}{}'.format('[{}]->'.format(params) if params else '', s)
            Lp = Lp.intersect(BasicSet(s))
        return Lp


    def ker(self, f):
        constraints = list(f.get_constraints())
        null_space = BasicSet.universe(self.space)
        constraint = Constraint.equality_alloc(self.space)

        for i, c in enumerate(constraints):
            for i in range(self.num_indices):
                constraint = constraint.set_coefficient_val(dim_type.out, i, -1 * c.get_coefficient_val(dim_type.in_, i))
            for i in range(self.num_params):
                constraint = constraint.set_coefficient_val(dim_type.param, i, c.get_coefficient_val(dim_type.param, i))
            constraint = constraint.set_constant_val(-1 * c.get_constant_val())
            null_space = null_space.add_constraint(constraint)

        return null_space


    def ker_from_map(self, f):
        if type(f) == str:
            f = BasicMap(f)
        mat = []
        for c in f.get_constraints():
            vars = c.get_var_dict()
            index_vals = [-1 * int(c.get_coefficient_val(v[0], v[1]).to_str()) for k, v in vars.items() if
                          v[0] != dim_type.param]
            mat.append(index_vals)
        mat = np.array(mat)

        indices = ','.join([str(i[0]) for i in self.indices])
        params = ','.join([str(i[0]) for i in self.params])
        constraints = []
        for c in mat:
            constraints.append('+'.join(['{}{}'.format(a[0], a[1]) for a in zip(c, indices.split(',') + params.split(','))]) + '=0')
        s = '{{[{}] : {}}}'.format(indices, ' and '.join(constraints))
        s = '{}{}'.format('[{}]->'.format(params) if params else '', s)
        return BasicSet(s)

    def get_facets(self, node=None):
        _node = node if node else self.root[self.chamber]
        return list(self.graph[self.chamber].neighbors(_node))

    def get_candidate_facets(self, node, fp):
        ret = []
        Lp = self.get_Lp(node)
        for facet in self.get_facets(node):
            if self.boundary_label(facet, fp, Lp) != BoundaryLabel.STRICT_BOUNDARY:
                ret.append(facet)
        return ret

    def get_weak_boundary_facets(self, node, fp):
        ret = []
        Lp = self.get_Lp(node)
        for facet in self.get_facets(node):
            if self.boundary_label(facet, fp, Lp) == BoundaryLabel.WEAK_BOUNDARY:
                ret.append(facet)
        return ret

    def ker_from_facet_normal(self, facet):
        mat = self.C[np.array(list(facet)), :self.num_indices]
        indices = ','.join([str(i[0]) for i in self.indices])
        params = ','.join([str(i[0]) for i in self.params])
        constraints = []
        for c in mat:
            constraints.append('+'.join(['{}{}'.format(a[0], a[1]) for a in zip(c, indices.split(',') + params.split(','))]) + '=0')
        s = '{{[{}] : {}}}'.format(indices, ' and '.join(constraints))
        s = '{}{}'.format('[{}]->'.format(params) if params else '', s)
        Lp = self.get_Lp(facet)
        return BasicSet(s).intersect(Lp)

    def decompose_projection_from_weak_facet(self, facet, fp, Lp):
        ker_c = self.ker_from_facet_normal(facet)
        ker_fp = self.ker_from_map(fp).intersect(Lp)
        I = ker_c.intersect(ker_fp)

        # TODO - this needs work
        # to make facet a strict boundary, we need a new fp1 where
        # ker(fp1) is a subset of ker(facet)

        num_params = len(self.params)
        I_no_params = I.project_out(dim_type.param, 0, num_params)
        if self.compute_rank(I_no_params) == 0:
            return None

        i_pieces = []
        for c in I.get_constraints():
            aff_str = c.get_aff().set_constant_val(0).to_str()
            start = aff_str.index('(') + 1
            end = aff_str.index(')')
            i_pieces.append(aff_str[start:end])

        p_pieces = [p[0].get_name() for p in self.params]

        fp_1 = '{}{{[{}]->[{}]}}'.format(
            '[{}]->'.format(','.join(p_pieces)) if num_params else '',
            ','.join([str(i[0]) for i in self.indices]),
            ','.join(i_pieces)
        )
        return fp_1

    # take an isl_point obj and parse it as a vector
    # given {Point}{ [1, 1, 0] }
    # return [1, 1, 0]
    def point_to_vec(self, isl_point):
        index_vals = [index[1][1] for index in self.indices]
        vec = [isl_point.get_coordinate_val(dim_type.out, i) for i in index_vals]
        return vec

    def label(self, facet, parent, rho):
        Lp = self.get_Lp(facet)
        c_mat = self.C[np.array(list(facet - parent)), :self.num_indices][0]
        rho_vec = self.point_to_vec(rho)
        orientation = np.matmul(c_mat, rho_vec)
        if orientation > 0:
            return 'inward'
        elif orientation == 0:
            return 'INV'
        else:
            return 'outward'

    def boundary_label(self, facet, fp, Lp):
        ker_c = self.ker_from_facet_normal(facet)
        ker_fp = self.ker_from_map(fp).intersect(Lp)
        I = ker_c.intersect(ker_fp)
        if self.compute_rank(ker_c) > 1 and self.compute_rank(ker_fp) > 1:
            if I.is_empty():
                return BoundaryLabel.NON_BOUNDARY
            else:
                if ker_fp.is_subset(ker_c):
                    return BoundaryLabel.STRICT_BOUNDARY
                else:
                    return BoundaryLabel.WEAK_BOUNDARY
        else:
            if ker_fp.is_subset(ker_c):
                return BoundaryLabel.STRICT_BOUNDARY
            else:
                return BoundaryLabel.NON_BOUNDARY


    def pretty_print(self):
        print('face-lattice:')
        for k_kfaces in reversed(list(self.kfaces[self.chamber].items())):
            k, kfaces = k_kfaces
            print('{}-faces: {}'.format(k, [set(facet) if facet else {} for facet in kfaces]))
        print()

    def print_rays(self, vertex_idx=None):
        print(self.get_rays(vertex_idx))

    def get_rays(self, vertex_idx):
        rays = self.rays[self.chamber][vertex_idx]
        return rays

    def print_vertex_exprs(self, vertex_idx=None):
        vertices = self.vertex_nodes[self.chamber][1]
        if vertex_idx:
            print(vertices[vertex_idx].get_expr())
        else:
            for i, v in enumerate(vertices):
                print(v.get_expr())

    def get_vertex_multi_affs(self):
        ret = set()
        for i, v in enumerate(self.vertex_nodes[self.chamber][1]):
            ret.add(CustomMultiAff(v.get_expr(), i, self.num_indices))
        return ret

    def get_vertex_affs(self, vertex_idx):
        multi_aff = self.vertex_nodes[self.chamber][1][vertex_idx].get_expr()
        def aff_to_vec(aff, num_params):
            return [aff.get_coefficient_val(dim_type.param, j).to_python() for j in range(num_params)] + [aff.get_constant_val().to_python()]

        return [aff_to_vec(multi_aff.get_aff(i), self.num_params) for i in range(self.num_params)]
