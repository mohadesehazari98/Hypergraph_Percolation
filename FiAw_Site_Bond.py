import networkx as nx
import random
from networkx.algorithms.approximation import steiner_tree
# ---------------------------------------------------------------------------------------------------------
# initialize the graph
def build_initial_grid(w, h):
    G = nx.Graph()
    for i in range(w):
        for j in range(h):
            G.add_node((i, j))
            parity = (i + j) % 2
            for dx in [-1, 1]:
                ni = i + dx
                if 0 <= ni < w:
                    G.add_edge((i, j), (ni, j))
            if parity == 0 and j + 1 < h:
                G.add_edge((i, j), (i, j + 1))
            elif parity == 1 and j - 1 >= 0:
                G.add_edge((i, j), (i, j - 1))
    return G
# ---------------------------------------------------------------------------------------------------------
# site-bond percolation
def percolate_graph(G, w, h, p, q, terminals):
    Gp = G.copy()
    # Bond 
    for edge in list(Gp.edges):
        if random.random() > p:
            Gp.remove_edge(*edge)
        else:
            Gp.edges[edge]['fidelity'] = 0.99
    # Site 
    for component in nx.connected_components(Gp):
        if terminals <= component:
            surviving_nodes = set()
            for node in component:
                if node in terminals or random.random() <= q:
                    surviving_nodes.add(node)
            return Gp.subgraph(surviving_nodes).copy()
    return None
# ---------------------------------------------------------------------------------------------------------
# calculate the tree
def compute_steiner_tree(G, terminals, weight='fidelity'):
    comp = nx.node_connected_component(G, next(iter(terminals)))
    if terminals <= comp:
        return steiner_tree(G.subgraph(comp).copy(), terminals, weight=weight, method="kou")
    return None
# ---------------------------------------------------------------------------------------------------------
# fidelity of the chain
def compute_fidelity_along_path(G, path):
    fidelity = G[path[0]][path[1]].get("fidelity", 1.0)
    for i in range(1, len(path) - 1):
        f_next = G[path[i]][path[i + 1]].get("fidelity", 1.0)
        fidelity = fidelity * f_next + ((1 - fidelity) * (1 - f_next)) / 3
    return fidelity
def analyze_steiner_tree_fidelity(Gp, T, terminals):
    candidates = [n for n in T if T.degree[n] == max(dict(T.degree()).values())]
    center = next((n for n in candidates if n in terminals), candidates[0])
    #print(f"Center node: {center}")
    fidelity_dict = {}
    for terminal in terminals:
        if terminal != center:
            path = nx.shortest_path(T, center, terminal)
            fidelity = compute_fidelity_along_path(Gp, path)
            fidelity_dict[terminal] = fidelity
        else:
            fidelity_dict[center] = 1.0
    return fidelity_dict
# ---------------------------------------------------------------------------------------------------------
# final fidelity
def compute_final_fidelity(F1, F2, F3):
    term1 = 1 / 6
    term2 = (F1 + F2 + F3) / 9
    term3 = (F1 * F2 + F1 * F3 + F2 * F3) * (2 / 9)
    fidelity_final = term1 - term2 + term3
    return fidelity_final
# ---------------------------------------------------------------------------------------------------------
# run 
# take these inputs from the user 
p = float(input("Please enter the success probability of bond generation p: "))
q = float(input("Please enter the success probability of entanglement swapping q: "))
w, h = map(int, input("Please enter the size of the network [w, h]: ").split(','))
N = int(input("Please enter the number of trials N: "))

A, B, C = (1, 1), (w - 2, h - 2), (w - 2, 1)
f_limit = 0.5
terminals = {A, B, C}
G = build_initial_grid(w, h)

fidelity_avg, cnt = [], 0

for _ in range(N):
    Gp = percolate_graph(G, w, h, p, q, terminals)
    if Gp and (T := compute_steiner_tree(Gp, terminals)):
        fidelity_dict = analyze_steiner_tree_fidelity(Gp, T, terminals)
        fidelity_ghz = compute_final_fidelity(*fidelity_dict.values())
        cnt += 1
        if fidelity_ghz > f_limit:
            fidelity_avg.append(fidelity_ghz)

count = cnt / N
rate = len(fidelity_avg) / N
average_fidelity = sum(fidelity_avg) / len(fidelity_avg) if fidelity_avg else 0.0

print(f"\nReturn: The total rate is = {count:.3f}")
print(f"Return: The rate above fidelity limit ({f_limit}) is = {rate:.3f}")
print(f"Return: The average fidelity (filtered) is = {average_fidelity:.3f}")



