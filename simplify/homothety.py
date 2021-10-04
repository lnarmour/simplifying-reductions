from islpy import *
import numpy as np


def vertices(bset, num_indices, num_params):
    chambers = []
    bset.polyhedral_hull().compute_vertices().foreach_cell(chambers.append)
    vertices = []
    for chamber in chambers:
        vertices.append([])
        chamber.foreach_vertex(vertices[-1].append)
        vertices[-1] = [v.get_expr() for v in vertices[-1]]
        # vertices are not guaranteed to be in same relative order
        # sort them in lexicographic order
        vertices[-1].sort(key=lambda v: multi_aff_to_vec(v, num_indices, num_params))
    return vertices


def subtract_maffs_i(l, r, i):
    if l and r:
        return l.get_aff(i) - r.get_aff(i)
    elif l and not r:
        return l.get_aff(i)
    elif not l and r:
        return r.get_aff(i).neg()


def build_map(set_space, multi_aff_0, multi_aff_1, nparam, n_out, direction=0):
    n_in = n_out
    space = Space.alloc(set_space.get_ctx(), nparam, n_in, n_out)
    for name, tuple in set_space.get_var_dict().items():
        type, pos = tuple
        space = space.set_dim_name(type, pos, name)
        if type == dim_type.out:
            space = space.set_dim_name(dim_type.in_, pos, name)

    m = BasicMap.universe(space)

    for i in range(n_in):
        if direction == 0:
            diff = subtract_maffs_i(multi_aff_1, multi_aff_0, i)
        else:
            diff = subtract_maffs_i(multi_aff_0, multi_aff_1, i)

        # can only set integer values so scale all dims by lcm denominators
        if nparam > 0:
            max_den = np.lcm.reduce([diff.get_coefficient_val(dim_type.param, j).get_den_val().to_python() for j in range(nparam)])
        else:
            max_den = diff.get_constant_val().get_den_val()
        constraint = Constraint.alloc_equality(space)
        constraint = constraint.set_coefficient_val(dim_type.out, i, max_den)
        constraint = constraint.set_coefficient_val(dim_type.in_, i, max_den * -1)
        for j in range(nparam):
            val = diff.get_coefficient_val(dim_type.param, j)
            constraint = constraint.set_coefficient_val(dim_type.param, j, val * max_den)
        constraint = constraint.set_constant_val(diff.get_constant_val() * max_den)
        m = m.add_constraint(constraint)
    assert len(m.get_constraints()) == n_in
    return m


def make_integer_valued(vec):
    max_den = np.lcm.reduce([v.get_den_val().to_python() for v in vec])
    return [(v * max_den).to_python() for v in vec]


def multi_aff_to_vec(maff, num_indices, num_params):
    # given maff: [((A1)N + (B1)M + ... + (C1)), ((A2)N + (B2)M + ... + (C2)), ... ]
    # return vec: [A1, B1, ..., C1, A2, B2, ..., C2, ... ]
    vec = []
    for i in range(num_indices):
        aff = maff.get_aff(i)
        vec += [aff.get_coefficient_val(dim_type.param, j) for j in range(num_params)] + [aff.get_constant_val()]
    return vec


def homothetic(s0, s1, chamber=0, verbose=None):
    def pprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    pprint('s0:', s0)
    pprint('s1:', s1)
    pprint()
    return bsets_are_homothetic(BasicSet(s0), BasicSet(s1), chamber=chamber, verbose=verbose)


def bsets_are_similar(bset0, bset1, num_indices):
    def S(bset):
        A = []
        for c in bset.get_constraints():
            k = []
            aff = c.get_aff()
            for i in range(num_indices):
                k.append(aff.get_coefficient_val(dim_type.in_, i))
            A.append(k)
        A.sort()
        B = []
        for a in A:
            B += a
        #print(B)
        return B

    return S(bset0) == S(bset1)


def bsets_are_homothetic(bset0, bset1, chamber=0, verbose=None):
    def pprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    space = bset0.get_space()
    num_indices = len([k for k, v in space.get_var_dict().items() if v[0] == dim_type.out])
    num_params = len([k for k, v in space.get_var_dict().items() if v[0] == dim_type.param])

    chambers_0 = vertices(bset0, num_indices, num_params)
    chambers_1 = vertices(bset1, num_indices, num_params)

    vertices_0 = chambers_0[chamber]
    vertices_1 = chambers_1[chamber]

    pprint('vertices_0:')
    [pprint(v) for v in vertices_0]
    pprint('vertices_1:')
    [pprint(v) for v in vertices_1]
    pprint()

    if len(vertices_0) != len(vertices_1):
        return False

    # vertices are sorted in increasing lexicographic order
    # translate bset1 so that it's min vertex lands on bset0's min vertex
    # the min vertex is then treated as the homothetic center
    lexmin_v0 = vertices_0[0]
    lexmin_v1 = vertices_1[0]
    m = build_map(space, lexmin_v0, lexmin_v1, num_params, num_indices)
    pprint('map:', m)
    pprint()
    translated_bset1 = bset1.apply(m)
    translated_chambers_1 = vertices(translated_bset1, num_indices, num_params)

    vertices_1 = translated_chambers_1[chamber]

    # now the min vertices are the same, take this as the homothetic center
    homothetic_center = lexmin_v0
    vertices_0 = vertices_0[1:]
    vertices_1 = vertices_1[1:]

    pprint('homothetic_center: ', homothetic_center)
    pprint()
    pprint('vertices_0:')
    [pprint(v) for v in vertices_0]
    pprint('translated_vertices_1:')
    [pprint(v) for v in vertices_1]
    pprint()

    # hstack vertex vectors relative to homothetic center
    def hstack_vertices(vertices):
        all_vecs = []
        for vertex in vertices:
            for i in range(num_indices):
                aff = vertex.get_aff(i) - homothetic_center.get_aff(i)
                all_vecs += [aff.get_coefficient_val(dim_type.param, j) for j in range(num_params)] + [aff.get_constant_val()]
        return all_vecs

    all_vecs_0 = make_integer_valued(hstack_vertices(vertices_0))
    all_vecs_1 = make_integer_valued(hstack_vertices(vertices_1))

    # to be homothetic, vec0 and vec must be linearly dependent
    mat = np.vstack((all_vecs_0, all_vecs_1))
    is_homothetic = np.linalg.matrix_rank(mat) == 1
    return is_homothetic


def homothety_tests():
    # negative cases
    assert not homothetic('{[i,j] : 0<=i,j<=10  }',
                          '{[i,j] : 0<=i<j<=200 }')
    assert not homothetic('{[i,j] : 17<=i<=217 and 15<=j<=155 }',
                          '{[i,j] : 0<=i<=10 and 0<=j<=8 }')
    assert not homothetic('[N,M]->{[i,j] : 0<=i<=N and 0<=j<=M }',
                          '[N,M]->{[i,j] : 0<=i<=2N and 0<=j<=4M }')
    assert not homothetic('[N,M]->{[i,j] : 0<=i<=N and 0<=j<=M and 3j>=i+3 }',
                          '[N,M]->{[i,j] : 0<=i<=2N and 0<=j<=2M and 3j>=i+3 }')
    assert not homothetic('[N]->{[i,j] : 0<=i,j and 3i-3N<=j<=i and 0<N }',
                          '[N]->{[i,j] : 0<=i,j and 3i-3N<=j<=i and i<=N }')

    # positive cases
    assert homothetic('{[i,j] : 0<=i<=10 and 0<=j<=7 }',
                      '{[i,j] : 0<=i<=200 and 0<=j<=140 }')
    assert homothetic('{[i,j] : 0<=i<=10 and 0<=j<=7 }',
                      '{[i,j] : 17<=i<=217 and 15<=j<=155 }')
    assert homothetic('{[i,j] : 17<=i<=217 and 15<=j<=155 }',
                      '{[i,j] : 0<=i<=10 and 0<=j<=7 }')
    assert homothetic('{[i,j,k,l] : 0<=i<=10 and 0<=j<=7 and -4<=k,l<=5 }',
                      '{[i,j,k,l] : 0<=i<=200 and 0<=j<=140 and -80<=k,l<=100 }')
    assert homothetic('[N,M]->{[i,j] : 0<=i<=N and 0<=j<=M and 3j>=i }',
                      '[N,M]->{[i,j] : 0<=i<=2N and 0<=j<=2M and 3j>=i }')
    assert homothetic('[N,M]->{[i,j] : 0<=i<=N and 0<=j<=M and 3j>=i+3 }',
                      '[N,M]->{[i,j] : 0<=i<=2N and 0<=j<=2M-1 and 3j>=i+3 }')
    assert homothetic('[N]->{[i,j,k] : 0<=i,j,k and i+j+k<=N }',
                      '[N]->{[i,j,k] : 0<=i,j,k and i+j+k<=13N }')
    assert homothetic('[N]->{[i,j] : 0<=i,j and 3i-3N<=j<=i and 0<N }',
                      '[N]->{[i,j] : 0<=i,j and 3i-3N<=j<=i and 0<N<=j }')