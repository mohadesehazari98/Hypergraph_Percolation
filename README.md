# Hypergraph_Percolation

## Synopsis
This repository contains simulations of **fidelity-aware site-bond percolation** in a grid-shaped quantum network. The network is arranged as a square grid with `w × h` nodes. Each node has unique coordinates `(i, j)` and a degree of 3 (three neighbors).

---

## Edge Connections
- Nodes with **parity = 0** are connected to their **upper neighbor** `(i, j+1)`.
- Nodes with **parity = 1** are connected to their **lower neighbor** `(i, j-1)`.
- The parity of a node is computed as:  
  ```
  parity = (i + j) % 2
  ```

---

## Site-Bond Percolation
- **Edges** survive with probability `p`. If successful, they are assigned a random fidelity `f`.
- **Nodes** survive (i.e., perform successful quantum swapping) with probability `q`.

---

## Entanglement Path
After the percolation process, we search for a **spanning tree** that includes all **terminals** (end nodes).  
- If multiple trees are possible, we compute the **Steiner tree**.  
- This resulting tree is called the **entanglement path**.

---

## Calculating the Final Fidelity
To compute the final fidelity, we map the entanglement path to the corresponding quantum processes.  
- Example: Each node on the branches of the tree performs a **Bell-state measurement**, while the root node performs a **multipartite projective measurement** to distribute the multipartite entangled state.  
- In the current implementation (3-qubit GHZ states), there are always three terminals and a single root.

---

## Calculating the Rate
We report two rates:  
1. **Raw rate** – number of successful multipartite entangled state distributions.  
2. **Filtered rate** – number of successful distributions where fidelity exceeds a predefined threshold, along with the **average fidelity** of these above-threshold events.

---

We are yet to omplete the hypergraph case 
