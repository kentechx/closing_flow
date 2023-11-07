import numpy as np
import scipy
import scipy.sparse as sp
import igl
import gpytoolbox

from .funcs import per_face_prin_curvature
from typing import Callable


def dihedral_angles(vs, fs):
    adj_tt, _ = igl.triangle_triangle_adjacency(fs)
    face_centers = igl.barycenter(vs, fs)
    face_normals = igl.per_face_normals(vs, fs, np.ones((len(fs), 3), dtype=vs.dtype) / 3.)

    # face dihedral angle
    face_dihedral_angles = np.arccos(
        np.clip((face_normals[:, None, :] * face_normals[adj_tt]).sum(-1), -1., 1.))  # (n, 3)
    face_dihedral_angles[adj_tt < 0] = 0
    face_dihedral_angles = (np.pi - face_dihedral_angles).clip(0, np.pi)
    is_convex = ((face_centers[adj_tt] - face_centers[:, None, :]) * face_normals[:, None, :]).sum(-1) < 1e-8  # (n, 3)
    face_dihedral_angles[~is_convex] = 2 * np.pi - face_dihedral_angles[~is_convex]
    return face_dihedral_angles


def discrete_mean_curvature(vs, fs):
    # see https://github.com/alecjacobson/gptoolbox/blob/7b83500838f31e764f62abe227de5edbdf77264f/mesh/discrete_mean_curvature.m
    A = dihedral_angles(vs, fs)
    l = igl.edge_lengths(vs, fs)[:, [2, 0, 1]]
    e, emap, _, _ = igl.edge_flaps(fs)
    emap = np.ascontiguousarray(emap.reshape(3, -1).T[:, [2, 0, 1]])  # column major!!
    # assert np.allclose(l, np.linalg.norm(vs[e[:, 0]] - vs[e[:, 1]], axis=1)[emap])

    adj_tt, _ = igl.triangle_triangle_adjacency(fs)
    A[adj_tt < 0] = np.pi  # make np.pi - A == 0

    cur = (0.5 * 0.5 * 0.5 * (np.pi - A) * l).reshape(-1)
    H = np.zeros(vs.shape[0])
    np.add.at(H, e[emap.reshape(-1), 0], cur)
    np.add.at(H, e[emap.reshape(-1), 1], cur)
    return H


def get_all_boundary_vids(fs):
    bs = igl.all_boundary_loop(fs)
    if len(bs) == 0:
        return np.array([], dtype=int)
    return np.concatenate(bs)


def discrete_gaussian_curvature(vs, fs):
    # see https://github.com/alecjacobson/gptoolbox/blob/7b83500838f31e764f62abe227de5edbdf77264f/mesh/discrete_gaussian_curvature.m
    K = np.zeros(len(vs))
    np.add.at(K, fs, igl.internal_angles(vs, fs))
    K = 2 * np.pi - K
    b = get_all_boundary_vids(fs)
    if len(b) > 0:
        K[b] = K[b] - np.pi
    return K


def incident_faces(fs: np.ndarray, vids: np.ndarray, v_incidence=2):
    selected = np.zeros(fs.shape[0], dtype=bool)
    selected[vids] = True
    return np.where(np.sum(selected[fs], axis=1) >= v_incidence)[0]


def closing_flow(V: np.ndarray, F: np.ndarray,
                 maxiter: int = 100,
                 dt: float = 1.,  # Time step
                 bd: float = 1 / 0.4,  # Bound of curvature
                 h: float = 0.05,  # Edge length
                 remesh_iterations: int = 1,
                 self_intersect: bool = False,
                 opening: bool = False,
                 always_recompute: bool = False,
                 is_active: Callable = None,
                 quadric_curvatures: bool = False,
                 tol: float = 1e-7,
                 ):
    """
    :param V: vertices of the input mesh
    :param F: faces of the input mesh
    :param maxiter: maximum number of iterations
    :param dt: time step of the flow
    :param bd: bound of curvature
    :param h: edge length of remeshing
    :param remesh_iterations: number of remeshing iterations
    :param self_intersect: whether to detect self-intersection faces
    :param opening: whether to do opening operation
    :param always_recompute: whether to recompute the active set at each iteration
    :param is_active: a function that takes the curvature as input and returns a boolean array indicating whether the
        vertex is active
    :param quadric_curvatures: whether to use quadric curvatures
    :param tol: tolerance of convergence
    :return:
        U: the output mesh vertices
        G: the output mesh faces
    """
    Vfull = V.copy()
    Ffull = F.copy()

    if is_active is None:
        is_active = lambda k: k < -bd
    recompute = True
    for iter in range(maxiter):
        Vprev = Vfull.copy()
        Fprev = Ffull.copy()

        #
        if iter == 1 or recompute:
            # compute the minimum curvature ki
            A = igl.adjacency_matrix(Ffull) + sp.eye(Vfull.shape[0], dtype=Ffull.dtype)
            M = igl.massmatrix(Vfull, Ffull)
            M.data = np.clip(M.data, 1e-8, np.inf)
            K = discrete_gaussian_curvature(Vfull, Ffull)
            H = discrete_mean_curvature(Vfull, Ffull)
            k = H[:, None] + np.array([[1, -1.]]) * np.tile(np.sqrt(H ** 2 - M * K + 0j)[:, None], (1, 2))

            if opening:
                K = -np.real(k[:, 0] / M.data)
            else:
                K = np.real(k[:, 1] / M.data)

            if quadric_curvatures:
                assert False

            # compute moving
            moving = np.where(is_active(K))[0]

            if not always_recompute:
                recompute = False

        if len(moving) == 0:
            # print("Active set is empty")
            break

        # 2-ring neighborhood
        active = np.where(A[moving, :].sum(0) > 0)[1]
        active = np.where(A[active, :].sum(0) > 0)[1]

        if always_recompute:
            active = np.arange(Vfull.shape[0])

        fid_active = incident_faces(Ffull, active)
        f_active = Ffull[fid_active]
        f_inactive = Ffull[np.setdiff1d(np.arange(len(Ffull)), fid_active)]
        if len(f_active) == 0:
            # print("Active set is empty")
            break

        v_active, f_active, I, J = igl.remove_unreferenced(Vfull, f_active)
        fixed_test = np.setdiff1d(np.arange(len(v_active)), I[moving])
        v_inactive, f_inactive, _, _ = igl.remove_unreferenced(Vfull, f_inactive)

        V = v_active
        F = f_active

        M = sp.diags(np.array(M[J, J])[0])
        dblA = sp.diags(igl.doublearea(V, F) / 2.)
        v_xyz = np.concatenate([V[:, 0], V[:, 1], V[:, 2]])

        boundary_verts = get_all_boundary_vids(F)
        fixed = np.unique(np.concatenate([fixed_test, boundary_verts]))
        fixed_xyz = np.concatenate([fixed, fixed + len(V), fixed + len(V) * 2])

        PD1, PD2, _, _ = per_face_prin_curvature(V, F)
        if opening:
            PD1 = PD2

        # solve
        proy_matrix = sp.hstack([sp.diags(PD1[:, 0]), sp.diags(PD1[:, 1]), sp.diags(PD1[:, 2])], format='csr')
        proyected_gradient = proy_matrix @ igl.grad(V, F)
        int_proy_grad_sq = -proyected_gradient.T @ dblA @ proyected_gradient
        Q = sp.block_diag([M, M, M], format='csr') - 0.01 * dt * sp.block_diag(
            [int_proy_grad_sq, int_proy_grad_sq, int_proy_grad_sq], format='csc')
        linear = -sp.block_diag([M, M, M], format='csr') @ v_xyz
        Aeq = scipy.sparse.csr_matrix((Q.shape[0], Q.shape[1]))
        Beq = np.zeros(Q.shape[0])
        uu = igl.min_quad_with_fixed(Q, linear[:, None], fixed_xyz, v_xyz[fixed_xyz], Aeq, Beq, True)[1]
        U = np.ascontiguousarray(uu.reshape(3, -1).T)

        # remesh
        U, F = gpytoolbox.remesh_botsch(U, F, remesh_iterations, h, True, fixed)
        # U = scipy.io.loadmat('/media/opening-and-closing-surfaces/figures/handles/U.mat')['U']
        # F = scipy.io.loadmat('/media/opening-and-closing-surfaces/figures/handles/F.mat')['F'].astype('i4') - 1
        U, F = map(np.ascontiguousarray, (U, F))
        Udup = U
        # Udup = U.copy()

        boundary_verts = get_all_boundary_vids(F)
        interior_verts = np.setdiff1d(np.arange(len(U)), boundary_verts)

        # Udup[interior_verts, :] = U[interior_verts] + 1e-8 * np.random.RandomState(42).rand(len(interior_verts), 3)

        # merge active and inactive surface
        Vfull = np.concatenate([v_inactive, Udup], axis=0)
        Ffull = np.concatenate([f_inactive, F + len(v_inactive)], axis=0)
        Vfull, SVI, SVJ, Ffull = igl.remove_duplicate_vertices(Vfull, Ffull, 0)
        interior_active_full = SVJ[interior_verts + len(v_inactive)]

        H_interior_active = discrete_mean_curvature(U, F)
        K_interior_active = discrete_gaussian_curvature(U, F)
        Mnew = sp.diags(np.ones(len(Vfull)), format='csr')
        M_active = igl.massmatrix(U, F)
        m_active = M_active.data
        # Mnew = Mnew + sp.diags(m_active[interior_verts] - 1, format='csr', shape=(len(Vfull), len(Vfull)))
        Mnew[interior_active_full, interior_active_full] += m_active[interior_verts] - 1
        M = Mnew

        k_interior_active = H_interior_active[:, None] + np.array([[1, -1.]]) * np.tile(
            np.sqrt(H_interior_active ** 2 - M_active * K_interior_active + 0j)[:, None], (1, 2))
        if opening:
            K_interior_active = -np.real(k_interior_active[:, 0] / M_active.data)
        else:
            K_interior_active = np.real(k_interior_active[:, 1] / M_active.data)

        if quadric_curvatures:
            PD1, PD2, PV1, PV2 = igl.principal_curvature(U, F)
            if opening:
                K_interior_active = -PV2
            else:
                K_interior_active = PV1

        moving = np.zeros((len(Vfull),))
        moving_interior_active = is_active(K_interior_active)

        moving[interior_active_full] = moving_interior_active[interior_verts]
        moving = np.where(moving)[0]

        E = SVJ[igl.edges(F + len(v_inactive))]
        A = sp.csr_matrix(
            (np.ones(len(E) * 2, dtype='i4'), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))),
            shape=(len(Vfull), len(Vfull)))
        A = A + sp.eye(len(Vfull), dtype='i4')

        if self_intersect and (iter + 1) % 10 == 0:
            assert False

        if (iter + 1) % 10 == 0:
            dist = igl.point_mesh_squared_distance(Vfull, Vprev, Fprev)[0]
            if dist.max() < tol:
                # print("Converged")
                break

    U, G = Vfull, Ffull
    return U, G
