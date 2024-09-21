import torch
import torch.nn.functional as F

def poisson_solver(f, u=None, iterations=500, omega=1.0, tol=1e-5):
    """
    Solve the Poisson equation ∇²u = f using the Gauss-Seidel method with Successive Over-Relaxation (SOR).

    :param f: Source term tensor of shape (batch, 1, height, width)
    :param u: Initial guess for u. If None, initializes to zeros.
    :param iterations: Maximum number of iterations to perform
    :param omega: Relaxation parameter for SOR (1.0 = Gauss-Seidel)
    :param tol: Tolerance for convergence based on residual
    :return: Solved potential tensor u of shape (batch, 1, height, width)
    """
    if u is None:
        u = torch.zeros_like(f)

    for it in range(iterations):
        # Pad u for easy indexing (Neumann boundary conditions)
        u_padded = F.pad(u, (1, 1, 1, 1), mode='replicate')

        # Sum of neighboring points
        sum_neighbors = (
            u_padded[:, :, :-2, 1:-1] +  # up
            u_padded[:, :, 2:, 1:-1] +   # down
            u_padded[:, :, 1:-1, :-2] +  # left
            u_padded[:, :, 1:-1, 2:]     # right
        )

        # Update rule with source term f
        u_new = 0.25 * (sum_neighbors - f)

        # Apply Successive Over-Relaxation (SOR)
        u_new = omega * u_new + (1 - omega) * u

        # Compute residual to check for convergence
        residual = torch.mean(torch.abs(u_new - u))
        if residual < tol:
            print(f"Converged at iteration {it+1} with residual {residual.item():.6f}")
            break

        u = u_new

    return u