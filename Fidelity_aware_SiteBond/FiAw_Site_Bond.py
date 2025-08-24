from Site_Bond_func import *
from tqdm import tqdm
import time, numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.colors import PowerNorm
from concurrent.futures import ProcessPoolExecutor

# run 
print("Hello! This code simulates site-bond percolation.")
print("The simulation uses the following parameters:")
print("  - p: The probability of successfully creating bell pairs between neighboring nodes.")
print("  - q: The probability of successfully performing projective measurements at a node.")
print("  - p_f: The probability of generating perfect elementary bell pairs (fidelity = 1).")
print("-" * 50)
try:
    w, h = map(int, input("Please enter the network size (width, height), e.g., '10,10': ").split(','))
    p_f = float(input("Please enter p_f (e.g., 0.7): "))
    f_limit= float(input("Please enter the fidelity cut-off threshold (a number between 0 and 1): "))
    N, res = map(int, input("Please enter the number of simulation trials N and the plot resolution: ").split(','))
except ValueError:
    print("\nInvalid input.")
# ---------------------------------------------------------------------------------------------------------
# the sweep function 
def _eval_pq(args):
    p, q, BASE, edges, idA, idB, idC, edge_index, p_f, f_limit, N, seed = args
    rng = np.random.default_rng()
    fidelity_avg = []
    cnt = 0
    for _ in range(N):
        A, edge_active, edge_fid_eff, node_mask = percolate_state(BASE, edges, p, q, p_f, idA, idB, idC, rng=rng)

        # terminals must be present after site step
        if not (node_mask[idA] and node_mask[idB] and node_mask[idC]):
            continue
        paths, _, _ = steiner_three_paths(A, idA, idB, idC, weighted=False)
        if paths is None:
            continue

        # map nodes->edge ids
        idxA = path_to_edge_indices(paths[0], edge_index)
        idxB = path_to_edge_indices(paths[1], edge_index)
        idxC = path_to_edge_indices(paths[2], edge_index)
        # DEBUG: when p_f == 1, every edge on the three arms must be exactly 1.0
        if p_f == 1.0:
            eids = np.concatenate([idxA, idxB, idxC])
            if eids.size and not np.allclose(edge_fid_eff[eids], 1.0, atol=0, rtol=0):
                print("pf=1 edge fidelity not 1.0:", edge_fid_eff[eids])

        # guard: every path edge must be active and have a defined fidelity
        def _ok(eids):
            fa = np.asarray(edge_fid_eff[eids], dtype=np.float64)
            return fa.size == 0 or (np.all(np.isfinite(fa)) and np.all(edge_active[eids]))

        if not (_ok(idxA) and _ok(idxB) and _ok(idxC)):
            # This indicates an A↔edge-mask mismatch. Treat as unsuccessful trial.
            continue
            
        fA = compute_fidelity_along_path(np.asarray(edge_fid_eff[idxA], dtype=np.float64))
        fB = compute_fidelity_along_path(np.asarray(edge_fid_eff[idxB], dtype=np.float64))
        fC = compute_fidelity_along_path(np.asarray(edge_fid_eff[idxC], dtype=np.float64))
        ghz_f = compute_final_fidelity(fA, fB, fC)

        # DEBUG: final fidelities must be exactly 1.0 when p_f == 1
        if p_f == 1.0 and not np.allclose([fA, fB, fC, ghz_f], 1.0, atol=0, rtol=0):
            print("pf=1 path/ghz not 1.0:", fA, fB, fC, ghz_f)

        cnt += 1
        if ghz_f >= f_limit:
            fidelity_avg.append(ghz_f)

    count = cnt / N
    rate = (len(fidelity_avg) / cnt) if cnt else 0.0
    average_fidelity = (sum(fidelity_avg) / len(fidelity_avg)) if fidelity_avg else 0.0
    return p, q, count, rate, average_fidelity
# ---------------------------------------------------------------------------------------------------------
def run_sweep(w, h, res, p_f, N, f_limit):
    # 0.95..1.00, includes 1.0
    params = np.linspace(0.7, 1.0, int(res)+1, endpoint=True)         
    G = build_initial_grid(w, h)

    # map nodes to 0..n-1
    node_to_id = {n: i for i, n in enumerate(G.nodes())}
    idA, idB, idC = map(node_to_id.get, [(1,1), (w-2,h-2), (w-2,1)])
    edges = np.array([(node_to_id[u], node_to_id[v]) for u, v in G.edges()], dtype=np.int32)
    edge_index = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edges)}

    # BASE as CSR (only shape is really needed if you rebuild A per job)
    M = len(edges)
    n = len(node_to_id)
    row = np.concatenate([edges[:,0], edges[:,1]])
    col = np.concatenate([edges[:,1], edges[:,0]])
    data = np.ones(2*M, dtype=np.float32)
    BASE = csr_matrix((data, (row, col)), shape=(n, n))

    # unique seeds per job (good hygiene for parallel RNG)
    seeds = np.random.SeedSequence().spawn(len(params)**2)
    jobs = [(p, q, BASE, edges, idA, idB, idC, edge_index, p_f, f_limit, N, int(seeds[k].generate_state(1)[0])) 
            for k, (p, q) in enumerate(((p, q) for p in params for q in params))]
    with ProcessPoolExecutor() as ex:
        results = list(tqdm(ex.map(_eval_pq, jobs, chunksize=4), total=len(jobs), desc="(p,q) grid"))

    count, rate, average_fidelity = [], [], []
    for _, _, c, r, af in results:
        count.append(c)
        rate.append(r)
        average_fidelity.append(af)
    return params, count, rate, average_fidelity
# ---------------------------------------------------------------------------------------------------------
start = time.perf_counter()
params, count, rate, average_fidelity = run_sweep(w, h, res, p_f, N, f_limit)
elapsed = time.perf_counter() - start
print(f"\nRun finished in {elapsed:.2f} seconds.")
m = len(params)
rate2d = np.clip(np.array(rate), 0, 1).reshape(m, m)
count2d = np.clip(np.array(count), 0, 1).reshape(m, m)
print(f"count range: {count2d.min():.4f}–{count2d.max():.4f}")
print(f"rate  range: {rate2d.min():.4f}–{rate2d.max():.4f}")
# ---------------------------------------------------------------------------------------------------------
# Figure 1 
m = len(params)
rate2d = np.array(rate).reshape(m, m)      # rows indexed by p, cols by q
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(rate2d.T, origin='lower', cmap='gray_r', extent=[params.min(), params.max(), params.min(), params.max()], norm=PowerNorm(gamma=0.45), aspect='auto') # previous = 0.55
cb = plt.colorbar(im, ax=ax, pad=0.02)
cb.set_label(f'GHZ Fidelity > {f_limit:.2f} Rate: [{im.norm.vmin:.3g}–{im.norm.vmax:.3g}]')
ax.set_xlabel('p  (Bell-pair success)')
ax.set_ylabel('q  (node measurement success)')
ax.set_title(f'Fidelity-aware Rate, {w*h} nodes\n' f'Perfect Bell pair probability: {p_f:.2f}')
plt.tight_layout()
plt.savefig(f"perc_rate_heatmap_pf_{w*h}.png", dpi=300)
plt.show()
# ---------------------------------------------------------------------------------------------------------
# Figure 2
m = len(params)
count2d = np.array(count).reshape(m, m)      # rows indexed by p, cols by q
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(count2d.T, origin='lower', cmap='gray_r', extent=[params.min(), params.max(), params.min(), params.max()], norm=PowerNorm(gamma=0.85), aspect='auto') # previous = 1.0
cb = plt.colorbar(im, ax=ax, pad=0.02)
cb.set_label(f'Raw Rate [{im.norm.vmin:.3g}–{im.norm.vmax:.3g}]')
ax.set_xlabel('p  (Bell-pair success)')
ax.set_ylabel('q  (node measurement success)')
ax.set_title(f'Network Size: {w*h} nodes')
plt.tight_layout()
plt.savefig(f"perc_count_heatmap_pf_{w*h}.png", dpi=300)
plt.show()


