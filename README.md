# Navier-Cauchy BEM

This is a minimal example demonstrating how to solve Navier-Cauchy
equation, i.e., linear elasticity, with mixed boundary condition using
BEM. The implementation is based on JAX.

![result](somig_exm.png)

## Linear Elasticity

In **linear elasticity**, the displacement field $u(x) \in R^d$
(with $d = 2,3$) satisfies the Navier-Cauchy equation:

$$
\mu \nabla^2 u + (\lambda + \mu) \nabla (\nabla \cdot u) = f \quad \text{in } \Omega
$$

where $\lambda, \mu$ are Lamé parameters and $f$ is a body force. The
boundary conditions are given as

- **Dirichlet (displacement):**  $u = g$ on $\Gamma_D$
- **Neumann (traction):** $\sigma(u) n = t$ on $\Gamma_N$

where $\sigma$ is the linear stress tensor, computed as $\mu(\nabla
u + \nabla u^T) + \lambda (\nabla \cdot u)I$.

---

## Boundary Integral Formulation

Instead of solving the PDE in the whole domain, BEM reformulates the
problem as an integral equation on the boundary $\Gamma =
\partial\Omega$. Assuming there is no body force, then for a point $x
\in \Omega$, its solution can be fully determined by the boundary
conditions through

$$
u_i(x) = \int_{\Gamma} U_{ij}(x,y) t_j(y)\, d\Gamma(y) - \int_{\Gamma} T_{ij}(x,y) u_j(y)\, d\Gamma(y), \quad x\in \Omega
$$

where $U_{ij}(x,y)$ is the fundamental solution (a.k.a., Kelvin
solution) of elasticity, $T_{ij}(x,y)$ is the traction kernel
associated with $U_{ij}$, $u_j(y)$ is the boundary displacement and
$t_j(y) = \sigma(u)(y) n(y)$ is the boundary traction.

---

## Boundary Integral Equation (BIE)

Taking the limit as $x \to \Gamma$ yields the integral equation we need to solve:

$$
c_{ij}(x) u_j(x) + \int_{\Gamma} T_{ij}(x,y) u_j(y)\, d\Gamma(y)
= \int_{\Gamma} U_{ij}(x,y) t_j(y)\, d\Gamma(y), \quad x \in \Gamma
$$

where $c_{ij}(x)$ depends on the local geometry (e.g., $c_{ij} = \frac{1}{2}\delta_{ij}$ at smooth boundary points).  

---

## Discretization

For simplicity, we use piecewise constant element for discretizing the
above BIE, which is evaluated at the barycenter of each element.

---