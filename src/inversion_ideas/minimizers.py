"""
Classes to define minimizers.
"""
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg
import torch

from .base import Minimizer, Objective
from .errors import ConvergenceWarning


class PrimalDual(Minimizer):
    """
        Computes the primal-dual solution for the auxiliary step in ADMM where the constraint is m = Zc
        and rows of Z lie in the probability simplex Δ^{K-1}. Uses Chambolle–Pock optimization.
    """

    def __init__(self, n_iter, classes=None, mesh=None, **kwargs):
        super().__init__(**kwargs)

        self.device = 'mps'
        self.tau = 0.5            # Primal step
        self.sigma = 0.5          # Dual step
        self.theta = 1.0          # Over-relaxation
        self.lambda_tv = 1e-2     # Weight on TV
        self.rho = 1.0            # Weight on coupling m = Zc
        self.n_iter = n_iter

        self.cell_vols = torch.tensor(mesh.cell_volumes, dtype=torch.float32, device=self.device)
        self.face_areas = torch.tensor(mesh.face_areas, dtype=torch.float32, device=self.device)

        self.cl = torch.tensor(classes[:, None], dtype=torch.float32, device=self.device)

    def __call__(
            self, 
            objective: Objective, 
            model: NDArray[np.float64], 
    ):
        """
            m: Tensor of shape (N, C)
            cl: Class centers of shape (K, C)
            Returns soft segmentation Z of shape (N, K)
        """

        # call the proximal operator to get variables for the the primal-dual
        f = objective(model)
        cell_vols = self.cell_vols
        face_areas = self.face_areas
        print("here")
        G_torch = objective.gradient(model)
        cl = self.cl

        K = len(cl)
        N = cell_vols.shape[0]

        # Initialize primal and dual variables
        model = torch.tensor(model, dtype=torch.float32, device=self.device)
        Z = torch.full((N, K), 1.0 / K, device=self.device)
        Y = torch.zeros((G_torch.shape[0], K), device=self.device)
        Z_bar = Z.clone()

        for it in range(self.n_iter):
            # --- Dual update (TV ascent) ---
            grad_Z_bar = torch.stack([G_torch @ Z_bar[:, k] for k in range(K)], dim=1)  # (nF, K)
            Y += self.sigma * grad_Z_bar

            Y_norm = torch.clamp(Y.norm(dim=1, keepdim=True), min=1.0)
            Y = Y / Y_norm  # project onto dual unit ball

            # --- Primal update (data + coupling + TV) ---
            div_Y = torch.stack([-G_torch.T @ Y[:, k] for k in range(K)], dim=1)  # (nC, K)
            Zc = Z @ cl                    # (N, C)
            residual = Zc.squeeze(-1) - model             # (N, C)
            grad_coupling = residual.squeeze(0).unsqueeze(-1) @ cl.T  # (N, K)

            Z_prev = Z.clone()
            Z -= self.tau * (f + self.rho * grad_coupling + self.lambda_tv * div_Y)

            # --- Projection onto simplex ---
            Z = self.project_rows_to_simplex(Z)

            # --- Over-relaxation ---
            Z_bar = Z + self.theta * (Z - Z_prev)

            if it % 2 == 0 or it == self.n_iter - 1:
                grad_Z = torch.stack([G_torch @ Z[:, k] for k in range(K)], dim=1)  # (nF, K)
                tv_energy = (face_areas[:, None] * grad_Z.norm(dim=1, keepdim=True)).sum()
                data_term = (Z * f * cell_vols[:, None]).sum()
                energy = data_term + self.lambda_tv * tv_energy
                print(f"Iter {it:03d} | Energy: {energy.item():.4f}")

        return Z.detach().cpu().numpy()
    
    def project_rows_to_simplex(self, Z):
        """
        Projects each row of Z onto the probability simplex Δ^{K-1}.
        """
        Z_sorted, _ = torch.sort(Z, descending=True, dim=1)  # (N, K)
        cumsum = torch.cumsum(Z_sorted, dim=1)               # (N, K)
        K = Z.shape[1]
        rho = torch.arange(1, K + 1, device=Z.device).float().view(1, -1)  # (1, K)

        t = (cumsum - 1) / rho  # (N, K)
        mask = (Z_sorted - t) > 0  # (N, K)

        # Count number of valid rho per row
        rho_star = mask.sum(dim=1, keepdim=True)  # (N, 1)

        # Gather theta using correct indices
        idx = (rho_star - 1).clamp(min=0)  # (N, 1)
        theta = torch.gather(t, 1, idx)  # (N, 1)

        return torch.clamp(Z - theta, min=0.0)

class ConjugateGradient(Minimizer):
    """
    Conjugate gradient minimizer.

    Parameters
    ----------
    cg_kwargs :
        Additional arguments to be passed to :func:`scipy.sparse.linalg.cg`.
    """

    def __init__(self, **cg_kwargs):
        self.cg_kwargs = cg_kwargs

    def __call__(
        self, objective: Objective, initial_model: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Minimize objective function with a Conjugate Gradient method.

        .. important::

            This minimizer should be used only for linear objective functions.

        Parameters
        ----------
        objective : Objective
            Objective function to be minimized.
        initial_model : (n_params) array
            Initial model used to start the minimization.

        Returns
        -------
        inverted_model : (n_params) array
           Inverted model obtained after minimization.

        Notes
        -----
        Minimize the objective function :math:`\phi(\mathbf{m})` by solving the system:

        .. math::

            \bar{\bar{\nabla}} \phi \mathbf{m}^{*} = - \bar{\nabla} \phi

        through a Conjugate Gradient algorithm, where :math:`\bar{\bar{\nabla}} \phi`
        and :math:`\bar{\nabla} \phi` are the the Hessian and the gradient of the
        objective function, respectively.
        """
        # TODO: maybe it would be nice to add a `is_linear` attribute to the objective
        # functions for the ones that generate a linear problem.
        gradient = objective.gradient(initial_model)
        hessian = objective.hessian(initial_model)
        model_step, info = cg(hessian, -gradient, **self.cg_kwargs)
        if info != 0:
            warnings.warn(
                "Conjugate gradient convergence to tolerance not achieved after "
                f"{info} number of iterations.",
                ConvergenceWarning,
                stacklevel=2,
            )
        inverted_model = initial_model + model_step
        return inverted_model
