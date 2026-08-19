"""Physics-informed ML for the full-mesh clot map (PHASE9).

Target: ``deploy_clot_score`` > 0.9 on the wall and > 0.7 off-wall, domain-restricted,
under the FIT/DEV protocol, given the GT flow field at t=0 as an input.

The physics that shapes every design choice here (docs/PHASE7_FINDINGS.md):

  * GT clot **is** ``{Mat >= 2e7}`` everywhere -- wall and off-wall, 0.0%/0.19% error.
    So the learning target is one continuous field and one threshold, not two arms.
  * Wall ``Mat`` is the exact time-integral of its own nodal derivative (rho 0.999), and
    COMSOL's own ``J0_Mat`` integrated locally ranks final wall ``Mat`` at 0.855.  The
    source law is right; it is the *inputs* (frozen gate, degraded ``d(sr,x)``) that are
    wrong.  So the physics belongs in the model as a **feature and a residual base**, not
    as something to replace.
  * Off-wall ``Mat`` is ~0.16x its owner wall node's, on one topological node row.
  * 75% of clot nucleates where the t=0 gate is already open; the other 25% is
    flow-mediated creep that a t=0 snapshot can only infer from context.
"""
