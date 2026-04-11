import torch
import torch.optim as optim

def predict_with_physics_correction(model, data, kernels, correction_steps=25, lr=1e-3):
    """
    Performs Inference-Time Physics Correction (ITPC) on a GINO-DEQ prediction.
    """
    device = data.x.device
    model.eval() # Ensure model weights are completely frozen

    # 1. The "Smart Guess": Let GINO-DEQ and Anderson find the base prediction
    with torch.no_grad():
        base_pred = model(data, solver="anderson")

    # 2. Setup Optimization: We optimize the PREDICTION TENSOR, not the model
    # Detach it from the model's computation graph and enable gradients
    pred_opt = base_pred.detach().clone()
    pred_opt.requires_grad_(True)

    # Use Adam (or L-BFGS) to tweak the flow field
    optimizer = optim.Adam([pred_opt], lr=lr)

    # Precompute WLS geometric properties ONCE
    props = kernels._get_geometric_props(data)

    for step in range(correction_steps):
        optimizer.zero_grad()

        # --- 3. Compute Physics Residuals on the CURRENT prediction ---
        # Momentum
        l_mom = kernels.navier_stokes_residual(pred_opt, data, props=props)

        # Continuity (Mass Conservation)
        c_u = kernels._compute_derivatives(pred_opt[:, 0:1], props)
        c_v = kernels._compute_derivatives(pred_opt[:, 1:2], props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        l_cont = kernels.continuity_loss(du_ij, data=data)

        # Boundary Conditions
        l_bc = kernels.boundary_condition_loss(pred_opt, data)
        l_io = kernels.inlet_outlet_loss(pred_opt, data)

        # 4. The Correction Loss
        # We heavily weight continuity to eliminate "leaky" mass flow
        loss = l_mom + (10.0 * l_cont) + (50.0 * l_bc) + (10.0 * l_io)

        loss.backward()
        optimizer.step()

        # 5. Hard-Enforcement (Optional but recommended)
        # Manually snap wall velocities exactly to zero after every step to prevent drift
        with torch.no_grad():
            mask_wall = data.mask_wall.view(-1).bool()
            if mask_wall.any():
                pred_opt[mask_wall, 0:2] = 0.0 # Hard no-slip condition

    return pred_opt.detach()