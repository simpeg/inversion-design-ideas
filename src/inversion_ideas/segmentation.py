import numpy as np
import torch
from inversion_ideas.base import Objective
from scipy.sparse import dia_array

class SegmentationNorm(Objective):
    """
        Evaluates $\sum_{i=1}^{N_{cells}} \sum_{j=1}^{N_{classes}} z_{ij} (m_i - c_j)^2$

        Segmentation norm
    
    """
    def __init__(self, Z, c, n_params=None):
        super().__init__()
        """
            Initialize the class with matrix Z and vector c.
            
            Parameters:
            Z (numpy.ndarray): A 2D array with shape (N_c, N_x * N_z).
            c (numpy.ndarray): A 1D array with length N_c.
        """
        self.Z = Z
        self.c = c
        self.N_c = c.shape[0]
        self.N_xz = Z.shape[1]
        self._n_params = n_params

    @property
    def n_params(self):
        return self._n_params
        
    def __call__(self, model):
        """
            Evaluate the function phi_{m,Z}.
            
            Parameters:
            m (numpy.ndarray): A 1D array with length N_x * N_z.
            
            Returns:
            float: The value of the function phi_{m,Z}.
        """
        
        return np.einsum('ij,ij->', self.Z, (model[:, None] - self.c)**2)

        
    def gradient(self, model):
        return 2*(self.Z.sum(axis=1)*model - self.Z @ self.c)
        
    def hessian(self, model):
        """
            Return the Hessian as a diagonal sparse array:
                H = 2 * (diag(s) ⊗ I_C)
            where s = sum_j Z_ij, so the scaling is per cell.
            For scalar m (C=1), this is the full diagonal.

            Returns
            -------
            H : scipy.sparse._dia.dia_array
                Diagonal sparse Hessian (N x N).
        """
        s = self.Z.sum(axis=1)  # (N,)
        diag_vals = 2.0 * s     # (N,)

        # Wrap as sparse diagonal
        return dia_array((diag_vals[np.newaxis, :], [0]), shape=(len(diag_vals), len(diag_vals)))

    

class SegmentationTotalVariation(Objective):
    """
        Sets up the kernel function for the primal-dual solution in the auxiliary step in ADMM 
        where the constraint is m = Zc
        and rows of Z lie in the probability simplex Δ^{K-1}. Uses Chambolle–Pock optimization.
    """

    def __init__(self, initial_model, c, mesh=None, n_params=None, **kwargs):
        super().__init__(**kwargs)
        self.device = 'mps'
        self.model = initial_model
        self.cl = c
        self.mesh = mesh
        self.Z = np.abs(np.random.randn(mesh.n_cells, len(c)))
        self.Z = (1 / self.Z.max(axis=1)) @ self.Z
        self._n_params = n_params

    def __call__(self, Z): # remove classes from call
        """
            m: Tensor of shape (N, C)
            cl: Class centers of shape (K, C)
            Returns soft segmentation Z of shape (N, K)
        """

        self.Z = Z
        mesh = self.mesh

        # Prepare tensors
        m = torch.tensor(self.model, dtype=torch.float32, device=self.device).unsqueeze(1)  # (N, C)
        cl = torch.tensor(self.cl, dtype=torch.float32, device=self.device)  # (K, C)
        m_exp = m.unsqueeze(1)      # (N, 1, C)
        cl_exp = cl.unsqueeze(0)    # (1, K, C)
        f = ((m_exp - cl_exp) ** 2).sum(dim=-1)  # (N, K)

        return f
    
    def gradient(self, model):

        mesh = self.mesh
        G = mesh.cell_gradient  # (nF, nC)
        return torch.tensor(G.toarray(), dtype=torch.float32, device=self.device)
        
    def hessian(self, model):
        pass

    def get_cell_face_areas(self):
        mesh = self.mesh
        return torch.tensor(mesh.face_areas, dtype=torch.float32, device=self.device)
    
    def get_cell_volumes(self):
        mesh = self.mesh
        return torch.tensor(mesh.cell_volumes, dtype=torch.float32, device=self.device)
    
    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params
