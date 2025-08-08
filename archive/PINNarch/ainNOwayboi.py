import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define the Neural Network Architecture
class PINN_TDSE(nn.Module):
    """
    A PINN for the 3D Time-Dependent Schrödinger Equation.
    The network takes (t, x, y, z) as input and outputs the real (u) and
    imaginary (v) parts of the wavefunction Psi.
    """
    def __init__(self):
        super(PINN_TDSE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 2)  # Two outputs for real (u) and imaginary (v) parts
        )

    def forward(self, t, x, y, z):
        inputs = torch.cat([t, x, y, z], dim=1)
        outputs = self.net(inputs)
        return outputs[:, 0:1], outputs[:, 1:2] # u, v

# 2. Define the Analytical Solution (Ground Truth)
def analytical_solution(t, x, y, z):
    """
    Analytical solution for a 3D Gaussian wave packet.
    Psi(x, y, z, t) = (1 + 4it)^(-3/2) * exp(-(x^2+y^2+z^2)/(1+4it))
    """
    denominator = 1 + 4j * t
    exponent = -(x**2 + y**2 + z**2) / denominator
    prefactor = denominator**(-1.5)
    psi = prefactor * np.exp(exponent)
    return psi.real, psi.imag

# 3. Define the PDE Loss Function
def pde_loss(model, t, x, y, z):
    """
    Calculates the residual of the 3D TDSE.
    The TDSE is split into its real and imaginary parts.
    f_real = u_t + v_xx + v_yy + v_zz
    f_imag = v_t - u_xx - u_yy - u_zz
    """
    u, v = model(t, x, y, z)

    # First derivatives w.r.t. time
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # Second derivatives w.r.t. space
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    # PDE residuals
    # Note: The TDSE is i*Psi_t = -Laplacian(Psi).
    # i*(u_t + i*v_t) = -(u_xx + u_yy + u_zz + i*(v_xx + v_yy + v_zz))
    # i*u_t - v_t = -u_xx - u_yy - u_zz - i*v_xx - i*v_yy - i*v_zz
    # Real part: -v_t = -u_xx - u_yy - u_zz  =>  v_t - u_xx - u_yy - u_zz = 0
    # Imaginary part: u_t = -v_xx - v_yy - v_zz => u_t + v_xx + v_yy + v_zz = 0
    f_real = u_t + v_xx + v_yy + v_zz
    f_imag = v_t - u_xx - u_yy - u_zz

    loss = nn.MSELoss()
    return loss(f_real, torch.zeros_like(f_real)) + loss(f_imag, torch.zeros_like(f_imag))


# 4. Generate Training Data
domain = [-1.5, 1.5] # Spatial domain for x, y, z
t_domain = [0.0, 0.2] # Time domain

N_initial = 5004
N_boundary = 5004
N_pde = 20000

# Initial condition points (t=0)
t_initial = torch.zeros(N_initial, 1, device=device)
x_initial = torch.tensor(np.random.uniform(domain[0], domain[1], (N_initial, 1)), dtype=torch.float32, device=device)
y_initial = torch.tensor(np.random.uniform(domain[0], domain[1], (N_initial, 1)), dtype=torch.float32, device=device)
z_initial = torch.tensor(np.random.uniform(domain[0], domain[1], (N_initial, 1)), dtype=torch.float32, device=device)
u_initial_true, v_initial_true = analytical_solution(t_initial.cpu().numpy(), x_initial.cpu().numpy(), y_initial.cpu().numpy(), z_initial.cpu().numpy())
u_initial_true = torch.tensor(u_initial_true, dtype=torch.float32, device=device)
v_initial_true = torch.tensor(v_initial_true, dtype=torch.float32, device=device)

# Boundary condition points (on the 6 faces of the cube for all t)
n_face = N_boundary // 6
t_boundary = torch.tensor(np.random.uniform(t_domain[0], t_domain[1], (N_boundary, 1)), dtype=torch.float32, device=device)
# Faces x=-1.5 and x=1.5
x_b0 = torch.full((n_face, 1), domain[0], device=device)
x_b1 = torch.full((n_face, 1), domain[1], device=device)
y_b01 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)
z_b01 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)
# Faces y=-1.5 and y=1.5
y_b2 = torch.full((n_face, 1), domain[0], device=device)
y_b3 = torch.full((n_face, 1), domain[1], device=device)
x_b23 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)
z_b23 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)
# Faces z=-1.5 and z=1.5
z_b4 = torch.full((n_face, 1), domain[0], device=device)
z_b5 = torch.full((n_face, 1), domain[1], device=device)
x_b45 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)
y_b45 = torch.tensor(np.random.uniform(domain[0], domain[1], (n_face, 1)), dtype=torch.float32, device=device)

x_boundary = torch.cat([x_b0, x_b1, x_b23, x_b23, x_b45, x_b45], dim=0)
y_boundary = torch.cat([y_b01, y_b01, y_b2, y_b3, y_b45, y_b45], dim=0)
z_boundary = torch.cat([z_b01, z_b01, z_b23, z_b23, z_b4, z_b5], dim=0)
u_boundary_true, v_boundary_true = analytical_solution(t_boundary.cpu().numpy(), x_boundary.cpu().numpy(), y_boundary.cpu().numpy(), z_boundary.cpu().numpy())
u_boundary_true = torch.tensor(u_boundary_true, dtype=torch.float32, device=device)
v_boundary_true = torch.tensor(v_boundary_true, dtype=torch.float32, device=device)

# PDE (collocation) points
t_pde = torch.tensor(np.random.uniform(t_domain[0], t_domain[1], (N_pde, 1)), dtype=torch.float32, device=device, requires_grad=True)
x_pde = torch.tensor(np.random.uniform(domain[0], domain[1], (N_pde, 1)), dtype=torch.float32, device=device, requires_grad=True)
y_pde = torch.tensor(np.random.uniform(domain[0], domain[1], (N_pde, 1)), dtype=torch.float32, device=device, requires_grad=True)
z_pde = torch.tensor(np.random.uniform(domain[0], domain[1], (N_pde, 1)), dtype=torch.float32, device=device, requires_grad=True)

# 5. Training the PINN
pinn_model = PINN_TDSE().to(device)
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()

epochs = 8000
start_time = time.time()

for epoch in range(epochs):
    pinn_model.train()

    # Initial Condition Loss
    u_initial_pred, v_initial_pred = pinn_model(t_initial, x_initial, y_initial, z_initial)
    loss_ic = mse_loss(u_initial_pred, u_initial_true) + mse_loss(v_initial_pred, v_initial_true)

    # Boundary Condition Loss
    u_boundary_pred, v_boundary_pred = pinn_model(t_boundary, x_boundary, y_boundary, z_boundary)
    loss_bc = mse_loss(u_boundary_pred, u_boundary_true) + mse_loss(v_boundary_pred, v_boundary_true)

    # PDE Residual Loss
    loss_pde_val = pde_loss(pinn_model, t_pde, x_pde, y_pde, z_pde)

    # Total Loss (can be weighted)
    total_loss = loss_ic + loss_bc + loss_pde_val

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}, PDE: {loss_pde_val.item():.6f}')

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# 6. Test and Visualize the Results
pinn_model.eval()

# Create a grid of test points on a slice (e.g., z=0) at a specific time (t=0.1)
grid_size = 50
t_test = 0.1
x = np.linspace(domain[0], domain[1], grid_size)
y = np.linspace(domain[0], domain[1], grid_size)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X) # Slice at z=0
T = np.full(X.shape, t_test)

# Prepare grid for torch
X_t = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
Y_t = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
Z_t = torch.tensor(Z.flatten(), dtype=torch.float32).view(-1, 1).to(device)
T_t = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

# Get PINN predictions and calculate probability density |Psi|^2 = u^2 + v^2
with torch.no_grad():
    u_pinn, v_pinn = pinn_model(T_t, X_t, Y_t, Z_t)
    prob_density_pinn = (u_pinn**2 + v_pinn**2).cpu().numpy().reshape(X.shape)

# Get analytical solution and its probability density
u_analytical, v_analytical = analytical_solution(T.flatten(), X.flatten(), Y.flatten(), Z.flatten())
prob_density_analytical = (u_analytical**2 + v_analytical**2).reshape(X.shape)

# Calculate absolute error
error = np.abs(prob_density_pinn - prob_density_analytical)

# Plotting
fig = plt.figure(figsize=(18, 5))
plt.suptitle(f"3D TDSE PINN vs. Analytical Solution (Probability Density at t={t_test}, z=0)", fontsize=16)

# Plot 1: PINN Prediction
ax1 = fig.add_subplot(1, 3, 1)
c1 = ax1.contourf(X, Y, prob_density_pinn, cmap='inferno', levels=50)
fig.colorbar(c1, ax=ax1)
ax1.set_title('PINN Prediction $|\Psi_{PINN}|^2$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal', adjustable='box')


# Plot 2: Analytical Solution
ax2 = fig.add_subplot(1, 3, 2)
c2 = ax2.contourf(X, Y, prob_density_analytical, cmap='inferno', levels=50)
fig.colorbar(c2, ax=ax2)
ax2.set_title('Analytical Solution $|\Psi_{true}|^2$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal', adjustable='box')


# Plot 3: Absolute Error
ax3 = fig.add_subplot(1, 3, 3)
c3 = ax3.contourf(X, Y, error, cmap='Reds', levels=50)
fig.colorbar(c3, ax=ax3)
ax3.set_title('Absolute Error')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal', adjustable='box')


plt.tight_layout(rect=[0, 0, 1, 0.96])
# Save the plot
plt.savefig('pinn_3d_tdse_comparison.png', dpi=300)
print("\nPlot saved as 'pinn_3d_tdse_comparison.png'")
# plt.show()