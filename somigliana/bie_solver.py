import jax
import jax.numpy as jnp
from loguru import logger

from .kelvinlet import *


jax.config.update("jax_enable_x64", True)


class BIESolver:
    """
    Solve the Cauchy-Navier equation
    """
    def __init__(self, boundary_nodes, boundary_elements, mu, nu):
        self.mu = mu
        self.nu = nu
        logger.critical(f"number of boundary elements={boundary_elements.shape[0]}")
        logger.critical(f"mu={self.mu}, nu={self.nu}")
        
        # boundary segment
        self.bnd_nodes_p = boundary_nodes[boundary_elements[:, 0], :]
        self.bnd_nodes_q = boundary_nodes[boundary_elements[:, 1], :]

        # number of quadratures per line element
        num_quadrature = 16
        self.gs, self.gw = gauss_quadrature_0_1(num_quadrature)
        
        def compute_geo(x1, x2):
            center = 0.5*(x1 + x2)
            tangent = x2 - x1
            length = jnp.linalg.norm(tangent)
            tangent /= length
            normal = jnp.array([tangent[1], -tangent[0]])
            return center, normal, length
        
        self.centers, self.normals, self.lengths = jax.vmap(compute_geo, in_axes=(0, 0))(self.bnd_nodes_p, self.bnd_nodes_q)

        # boundary integral matrices, displacement and traction
        self.T_b2b = None
        self.K_b2b = None
        
        # unknown variables to solve
        self.bdy_uh = None
        self.bdy_qh = None

    def num_boundary_elements(self):
        return self.centers.shape[0]
        
    def assemble_kernel_matrices(self):
        """
        Assemble T and K matrices for boundary integral equation
        c(xi) * u(xi) + ∫ T*(xi,x) u(x) dΓ = ∫ K*(xi,x) q(x) dΓ
        """
        
        # boundary traction matrix
        inner_vmap = jax.vmap(integrate_T_over_element, in_axes=(None, 0, 0, 0, None, None, None, None))
        T_vmap = jax.vmap(inner_vmap, in_axes=(0, None, None, None, None, None, None, None))
        self.T_b2b = T_vmap(self.centers, self.bnd_nodes_p, self.bnd_nodes_q, self.normals, self.mu, self.nu, self.gs, self.gw)
        logger.info(f"T shape={self.T_b2b.shape}")
        logger.info(f"PoU violation={jnp.max(jnp.abs(jnp.sum(self.T_b2b, axis=1)))}")

        # boundary displacement matrix
        inner_vmap = jax.vmap(integrate_K_over_element, in_axes=(None, 0, 0, None, None, None, None))
        K_vmap = jax.vmap(inner_vmap, in_axes=(0, None, None, None, None, None, None))
        self.K_b2b = K_vmap(self.centers, self.bnd_nodes_p, self.bnd_nodes_q, self.mu, self.nu, self.gs, self.gw)
        logger.info(f"K shape={self.K_b2b.shape}")
        
    def solve(self, boundary_condition):
        bc_types = boundary_condition['type']
        bc_values = boundary_condition['value']
        
        # record types of boundary conditions
        is_dirichlet = bc_types == 0
        is_neumann = bc_types == 1
        
        # BIE left-hand side
        LHS = jnp.zeros_like(self.K_b2b)
        LHS = LHS.at[:, is_dirichlet].set(-self.K_b2b[:, is_dirichlet])
        LHS = LHS.at[:, is_neumann  ].set( self.T_b2b[:, is_neumann  ])
        
        # BIE right-hand side
        rhs = -jnp.tensordot(self.T_b2b[:, is_dirichlet], bc_values[is_dirichlet], axes=([1, 3], [0, 1])) + jnp.tensordot(self.K_b2b[:, is_neumann], bc_values[is_neumann], axes=([1, 3], [0, 1]))

        # solve BIE
        n = LHS.shape[0]
        solution_vector = jnp.linalg.solve(LHS.transpose(0, 2, 1, 3).reshape(2*n, 2*n), rhs.ravel()).reshape(bc_values.shape)

        # full boundary conditions
        bdy_uh = jnp.zeros_like(bc_values)
        bdy_qh = jnp.zeros_like(bc_values)
        bdy_uh = bdy_uh.at[is_dirichlet].set(bc_values[is_dirichlet])
        bdy_uh = bdy_uh.at[is_neumann  ].set(solution_vector[is_neumann])
        bdy_qh = bdy_qh.at[is_dirichlet].set(solution_vector[is_dirichlet])
        bdy_qh = bdy_qh.at[is_neumann  ].set(bc_values[is_neumann])
        
        self.bdy_uh = bdy_uh
        self.bdy_qh = bdy_qh
                        
    def evaluate_solution(self, eval_points):
        n_eval = eval_points.shape[0]
        logger.info(f"n_eval={n_eval}")

        # T(x, y)
        inner_vmap = jax.vmap(integrate_T_over_element, in_axes=(None, 0, 0, 0, None, None, None, None))
        T_vmap = jax.vmap(inner_vmap, in_axes=(0, None, None, None, None, None, None, None))
        T = T_vmap(eval_points, self.bnd_nodes_p, self.bnd_nodes_q, self.normals, self.mu, self.nu, self.gs, self.gw)

        # K(x, y)
        inner_vmap = jax.vmap(integrate_K_over_element, in_axes=(None, 0, 0, None, None, None, None))
        K_vmap = jax.vmap(inner_vmap, in_axes=(0, None, None, None, None, None, None))
        K = K_vmap(eval_points, self.bnd_nodes_p, self.bnd_nodes_q, self.mu, self.nu, self.gs, self.gw)

        return -jnp.tensordot(T, self.bdy_uh, axes=([1, 3], [0, 1])) + jnp.tensordot(K, self.bdy_qh, axes=([1, 3], [0, 1]))