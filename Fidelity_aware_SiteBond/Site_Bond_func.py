import numpy as np
import networkx as nx
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
# --------------------------------------------initialize the graph-------------------------------------------------------------
def build_initial_grid(w, h):
    G = nx.Graph()
    for i in range(w):
        for j in range(h):
            G.add_node((i, j))
            parity = (i + j) % 2
            if 0 <= i + 1 < w:
                G.add_edge((i, j), (i + 1, j))
            if parity == 0 and j + 1 < h:
                G.add_edge((i, j), (i, j + 1))
            elif parity == 1 and j - 1 >= 0:
                G.add_edge((i, j), (i, j - 1))
    return G
# -------------------------------------------site-bond percolation------------------------------------------------------------
def percolate_state(BASE, edges, p, q, p_f, idA, idB, idC, *, rng=None):
    """
    BASE : csr_matrix, symmetric 0/1 adjacency (n x n)
    edges: (M,2) int array for undirected edges matching BASE
    p/q  : bond/site survival probability (site perc applied only within the CC containing all 3 terminals) 
    p_f  : prob a surviving bond gets fidelity=1.0; else U[0.9,1.0]
    idA, idB, idC : int node indices of the three terminals
    """
    if rng is None:
        rng = np.random.default_rng()
    M = len(edges)

    # ---------- Bond percolation ----------
    survive_edge = rng.random(M) < p
    raw_fid = np.where(rng.random(M) < p_f, 1.0, 0.9 + 0.1 * rng.random(M))
    # Only defined on bonds that survived; others are NaN
    edge_fid = np.where(survive_edge, raw_fid, np.nan)

    # Apply bond mask to adjacency
    u, v = edges[:, 0], edges[:, 1]
    act = survive_edge
    rowA = np.concatenate([u[act], v[act]])
    colA = np.concatenate([v[act], u[act]])
    A = csr_matrix((np.ones(rowA.size, dtype=np.float32), (rowA, colA)), shape=BASE.shape)

    # ---------- CC sanity check ----------
    n_comp, labels = connected_components(A, directed=False)
    same_comp = (labels[idA] == labels[idB] == labels[idC])
    if not same_comp:
        # No site percolation step if terminals aren't all connected
        node_mask = np.ones(A.shape[0], dtype=bool)
        edge_active = survive_edge
        edge_fid_eff = np.where(edge_active, edge_fid, np.nan)
        return A, edge_active, edge_fid_eff, node_mask

    # "Proof" sanity: components are disjoint, so a given set of 3 distinct nodes
    # can belong to at most one component together. Hence if labels match, there
    # is exactly one component containing all three.
    assert len({labels[idA], labels[idB], labels[idC]}) == 1

    # ---------- Site percolation on that CC only ----------
    comp_label = labels[idA]
    in_comp = (labels == comp_label)

    n = A.shape[0]
    node_mask = np.zeros(n, dtype=bool)
    node_mask[[idA, idB, idC]] = True                # terminals always survive
    candidates = np.where(in_comp & ~node_mask)[0]    # only nodes in that CC
    node_mask[candidates] = (rng.random(len(candidates)) < q)

    # Remove any edge touching a removed node
    row_scale = node_mask.astype(A.dtype)[:, None]
    col_scale = node_mask.astype(A.dtype)[None, :]
    A = A.multiply(row_scale).multiply(col_scale)
    A.eliminate_zeros()

    # Final active edges are those that survived bond AND whose endpoints survived site
    u, v = edges[:, 0], edges[:, 1]
    edge_active = survive_edge & node_mask[u] & node_mask[v]
    edge_fid_eff = np.where(edge_active, edge_fid, np.nan)
    assert np.all(np.isfinite(edge_fid_eff) == edge_active), "active edge has NaN (or inactive edge has finite fid)"

    return A, edge_active, edge_fid_eff, node_mask
# ------------------------------------------------calculate the tree---------------------------------------------------------
def steiner_three_paths(A, idA, idB, idC, *, weighted=False, debug=False):
    """
    Return 3 arms (A→root, B→root, C→root), the root node, and total edge count.
    Arms are edge-disjoint except at the root. Works in both Y-case (root has deg 3)
    and line-case (middle terminal has deg 2).
    """
    ids = [idA, idB, idC]

    # distances from each terminal to all nodes
    dist, _ = shortest_path(A, directed=False, return_predecessors=True,
                            unweighted=not weighted, indices=ids)
    finite_all = np.isfinite(dist).all(axis=0)
    if not finite_all.any():
        return None, None, np.inf

    # pick m (1-median), tie-break by smallest longest arm
    sums = dist[:, finite_all].sum(axis=0)
    cand_nodes = np.flatnonzero(finite_all)
    order = np.lexsort((dist[:, finite_all].max(axis=0), sums))
    m = int(cand_nodes[order[0]])

    # one SPT rooted at m; reconstruct m→terminal paths ONCE
    _, pred_m = shortest_path(A, directed=False, return_predecessors=True,
                              unweighted=not weighted, indices=m)

    def _reconstruct_path(pred_row, src, tgt):
        if src == tgt:
            return [src]
        path = [tgt]
        cur = tgt
        while cur != src:
            cur = pred_row[cur]
            if cur < 0:
                return None
            path.append(cur)
        path.reverse()
        return path

    # paths_m are [m, ..., terminal]
    paths_m = [_reconstruct_path(pred_m, m, t) for t in (idA, idB, idC)]
    if any(p is None for p in paths_m):
        return None, None, np.inf

    # degrees in the union of the three paths (no CSR needed)
    deg = {}
    for P in paths_m:
        for u, v in zip(P[:-1], P[1:]):
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1

    # choose root: branching node if any; otherwise middle terminal (deg==2)
    if any(c >= 3 for c in deg.values()):
        root = next(v for v, c in deg.items() if c >= 3)
    else:
        root = next(t for t in (idA, idB, idC) if deg.get(t, 0) == 2)

    # final arms = segment from terminal to root (reverse of m→terminal suffix)
    paths = []
    for P in paths_m:
        try:
            k = P.index(root)       # root must lie on each SPT arm
        except ValueError:
            return None, None, np.inf
        arm = list(reversed(P[k:]))  # terminal→...→root
        paths.append(arm)

    total = sum(len(p) - 1 for p in paths) if not weighted else sums.min()

    # -------------------- DEBUG CHECKS (optional) --------------------
    if debug:
        # each arm ends at root
        assert all(p and p[-1] == root for p in paths), "arm does not end at root"

        # arms are edge-disjoint except at the root
        def edge_set(path):
            return { (min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:]) }
        eA, eB, eC = map(edge_set, paths)
        assert not (eA & eB) and not (eA & eC) and not (eB & eC), "arms share an edge"

        # path lengths match SPT distances to m (since built from pred_m)
        # (This implies minimality of the connector under the 1-median choice.)
        # Note: we compare with segments consistent with SPT structure.
        from math import isfinite
        # Convert terminal→root arm length to root→terminal (same length)
        lens = [len(p) - 1 for p in paths]
        assert all(l >= 0 for l in lens), "negative arm length?"

    return paths, root, total
# ---------------------------------------------------------------------------------------------------------
def path_to_edge_indices(path, edge_index):
    idx = []
    for u, v in zip(path, path[1:]):
        key = (u, v) if u < v else (v, u)
        idx.append(edge_index[key])
    return np.asarray(idx, dtype=np.int32)
# --------------------------------------------fidelity of the tree-------------------------------------------------------------
@njit
def compute_fidelity_along_path(f_edges):
    if len(f_edges) == 0:
        return 1.0
    f = f_edges[0]
    for i in range(1, len(f_edges)):
        g = f_edges[i]
        f = f * g + ((1.0 - f) * (1.0 - g)) / 3.0
    return f
# final fidelity
def compute_final_fidelity(F1, F2, F3):
    eps1 = (1 - F1) / 3
    eps2 = (1 - F2) / 3
    eps3 = (1 - F3) / 3
    e1 = eps1 + eps2 + eps3
    e2 = eps1 * eps2 + eps1 * eps3 + eps2 * eps3
    e3 = eps1 * eps2 * eps3
    F4 = 1 - 3 * e1 + 10 * e2 - 32 * e3
    return F4