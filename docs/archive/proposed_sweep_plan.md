# Architectural Pivots: Off-Wall Clot Growth Sweep

This document outlines the implementation plan to resolve the GNN "firewall" effect (Hop 1 shear gap) and the static distance-to-wall ceiling (Hop 3+ ceiling), enabling GNN rollouts to predict clots forming in the interior lumen.

---

## 1. Near-Wall Boundary Layer Fixes (Hop 1 Gap)

### Pivot 1: Decoupled Linear-Subgraph Message Passing (Skip-Hop GNN)
* **Goal:** Bypassing the zero-value Hop 1 mid-side nodes during GNN convolution layers to prevent the GNN from learning a propagation blockade.
* **Implementation Details:**
  1. Identify corner nodes (even hops from the wall: 0, 2, 4, ...) versus mid-side nodes (odd hops: 1, 3, 5, ...).
  2. Restrict the GNN's `SAGEConv` layers to operate only on the node-induced subgraph of **even-hop (corner) nodes**.
  3. Reconstruct species concentration values for mid-side nodes (odd hops) downstream using quadratic interpolation of their adjacent corner nodes.
  4. Ensure this subgraph division is re-evaluated per training/inference sample at the start of unrolling.

### Pivot 2: Differentiable Readout Shear Gate
* **Goal:** Removing the burden of learning the sharp high-shear boundary gradient from the GNN convolutions.
* **Implementation Details:**
  1. Let the GNN predict a smooth spatial species concentration profile $P_{Mat}$ across the entire graph.
  2. Implement a readout gate multiplied directly by the GNN's raw prediction:
     $$Mat_{pred} = P_{Mat} \odot \sigma\left(\frac{lss - \dot{\gamma}}{\tau}\right)$$
     where $lss$ is the local shear threshold, $\dot{\gamma}$ is the local shear rate, and $\tau$ is a soft temperature parameter (e.g., $1.0$, scaling down for sharpness).
  3. Keep the temperature $\tau$ soft enough during backpropagation to prevent vanishing gradients.

---

## 2. Lumen Propagation Fixes (Hop 3+ Ceiling)

### Pivot 3: Dynamic Geometry Occlusion Loop (Flow Re-Solving)
* **Goal:** Diverting fluid flow dynamically around growing clots to shift stagnation zones outward.
* **Implementation Details:**
  1. During the unrolling loop (rollout), at step $t$, identify nodes where predicted clot fraction $\phi \ge 0.5$.
  2. Append these clotted nodes to the solid wall mask.
  3. Recompute the Signed Distance Field (SDF) and zero out the flow velocities $\vec{u}$ inside the clotted nodes.
  4. Periodically (e.g., every 5 unrolling macro steps) trigger a GINO-DEQ forward pass to re-solve the flow field with the updated geometry boundaries.

### Pivot 4: Autocatalytic Dynamic Frontier Growth
* **Goal:** Enabling the clot to grow outward as a self-sustaining front rather than being capped by static distance-to-wall features.
* **Implementation Details:**
  1. Restrict the active GNN prediction scope to a 1-hop dynamic frontier around already-committed clot nodes.
  2. Model clot growth rate at the frontier as a reaction-diffusion boundary kinetics term driven by local bulk platelet (AP) and thrombin (T) convective fluxes hitting the front.

---

## 3. Sweep Configuration Adjustments
1. **Disable Wall-Only Constraints:** Ensure `"CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0"` is set in the sweep config.
2. **Increase Nucleation Hops:** Set `"CLOT_V2_NUCLEATION_HOPS": "3"` to allow the dynamic nucleation mask to accommodate Hop 2 and Hop 3 nodes when the GNN firewall is resolved.
