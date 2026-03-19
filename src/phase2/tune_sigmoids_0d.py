import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. SOFT LOGIC FUNCTIONS
# ==========================================
def soft_step(x, threshold, temperature):
    """Smooth approximation of if (x > threshold)"""
    return torch.sigmoid((x - threshold) / temperature)

def hard_step(x, threshold):
    """Rigid COMSOL logic"""
    return torch.where(x > threshold, 1.0, 0.0)

# ==========================================
# 2. TUNING PARAMETERS (CHANGE THESE!)
# ==========================================
# Objective: Make the dashed orange line (Soft) hug the solid blue line (Hard)
# without becoming a perfect 90-degree angle.

# A. Chemical Activation (Omega > 1.0)
omega_thresh = 1.0
T_omega = 0.05       # <-- Tune this

# B. Mechanical Activation (Shear > 1000)
shear_thresh = 1000.0
T_shear = 50.0       # <-- Tune this

# C. Platelet Viscosity Ramp (Mat > 2e7)
mat_thresh = 2e7
T_mat = 1e6          # <-- Tune this

# D. Fibrin Viscosity Ramp (FI > 0.6)
fi_thresh = 0.6
T_fi = 0.05          # <-- Tune this

# ==========================================
# 3. GENERATE DOMAINS (Tensors)
# ==========================================
omega_vals = torch.linspace(0, 3, 500)
shear_vals = torch.linspace(0, 2000, 500)
mat_vals = torch.linspace(0, 5e7, 500)
fi_vals = torch.linspace(0, 2, 500)

# ==========================================
# 4. PLOTTING SCRIPT
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("0D Sandbox: Soft Logic Gradient Tuning", fontsize=16)

# Plot 1: Chemical Activation
axs[0, 0].plot(omega_vals, hard_step(omega_vals, omega_thresh), 'b-', lw=2, label="COMSOL (Hard)")
axs[0, 0].plot(omega_vals, soft_step(omega_vals, omega_thresh, T_omega), 'r--', lw=2, label=f"PyTorch (Temp={T_omega})")
axs[0, 0].set_title("Chemical Activation (Omega)")
axs[0, 0].set_xlabel("Omega")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Plot 2: Mechanical Activation
axs[0, 1].plot(shear_vals, hard_step(shear_vals, shear_thresh), 'b-', lw=2, label="COMSOL (Hard)")
axs[0, 1].plot(shear_vals, soft_step(shear_vals, shear_thresh, T_shear), 'r--', lw=2, label=f"PyTorch (Temp={T_shear})")
axs[0, 1].set_title("Mechanical Activation (Shear Rate)")
axs[0, 1].set_xlabel("Shear Rate (1/s)")
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()

# Plot 3: Platelet Viscosity Multiplier (mu1)
axs[1, 0].plot(mat_vals, hard_step(mat_vals, mat_thresh), 'b-', lw=2, label="COMSOL (Hard)")
axs[1, 0].plot(mat_vals, soft_step(mat_vals, mat_thresh, T_mat), 'r--', lw=2, label=f"PyTorch (Temp={T_mat:.1e})")
axs[1, 0].set_title("Platelet Viscosity Ramp (Mat)")
axs[1, 0].set_xlabel("Activated Platelets (Mat)")
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend()

# Plot 4: Fibrin Viscosity Multiplier (mu2)
axs[1, 1].plot(fi_vals, hard_step(fi_vals, fi_thresh), 'b-', lw=2, label="COMSOL (Hard)")
axs[1, 1].plot(fi_vals, soft_step(fi_vals, fi_thresh, T_fi), 'r--', lw=2, label=f"PyTorch (Temp={T_fi})")
axs[1, 1].set_title("Fibrin Viscosity Ramp (FI)")
axs[1, 1].set_xlabel("Fibrin Concentration (FI)")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()

plt.tight_layout()
plt.show()