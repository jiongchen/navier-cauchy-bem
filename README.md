# Navier-Cauchy BEM

This is a minimal example demonstrating how to solve Navier-Cauchy
equation, i.e., linear elasticity, with mixed boundary condition using
BEM. The implementation is based on JAX.

![result](somig_exm.png)

## Linear Elasticity

In linear elasticity, the displacement field $u(x) \in R^d$
(with $d = 2,3$) satisfies the **Navier-Cauchy equation**:

$$
\mu \nabla^2 u + (\lambda + \mu) \nabla (\nabla \cdot u) = f \quad \text{in } \Omega
$$

where $\lambda$ and $\mu$ are the Lamé constant, and $f$ is a body force. The
boundary conditions are prescribed as:

- **Dirichlet (displacement):**  $u = g$ on $\Gamma_D$
- **Neumann (traction):** $\sigma(u) n = t$ on $\Gamma_N$

where $\sigma$ is the linear stress tensor, given by

$$
\sigma(u) = \mu(\nabla u + \nabla u^T) + \lambda (\nabla \cdot u)I.
$$

---

## Boundary Integral Formulation

The Boundary Element Method reformulates the PDE as an integral
equation on the boundary $\Gamma = \partial\Omega$, avoiding the need
to solve the PDE in the full domain. Assuming no body force, the
displacement at an interior point $x \in \Omega$ can be expressed
entirely in terms of the boundary data:

$$
u_i(x) = \int_{\Gamma} U_{ij}(x,y) t_j(y) \mathrm{d}\Gamma(y) - \int_{\Gamma} T_{ij}(x,y) u_j(y) \mathrm{d}\Gamma(y), \quad x\in \Omega
$$

where $U_{ij}(x,y)$ is the fundamental (Kelvin) solution of linear
elasticity, $T_{ij}(x,y)$ is the traction kernel associated with
$U_{ij}$, $u_j(y)$ is the boundary displacement, and $t_j(y) =
\sigma(u)(y) n(y)$ is the boundary traction. This identity is also
known as the **Somigliana identity**, which is the basis for our
derivation of [Somigliana
Coordinates](https://jiongchen.github.io/files/somi-paper.pdf), a type
of matrix-valued generalized barycentric coordinates for cage
deformation.

---

## Boundary Integral Equation (BIE)

Considering the jump relation of the **double-layer potential** across
the boundary, we arrive at the boundary integral equation to solve for
the boundary traction $t$ on $\Gamma_D$ and displacement $u$ on
$\Gamma_N$, through

$$
c_{ij}(x) u_j(x) + \int_{\Gamma} T_{ij}(x,y) u_j(y) \mathrm{d}\Gamma(y)
= \int_{\Gamma} U_{ij}(x,y) t_j(y) \mathrm{d}\Gamma(y), \quad x \in \Gamma
$$

where $c_{ij}(x)$ depends on the local geometry (e.g., $c_{ij} =
\frac{1}{2}\delta_{ij}$ at smooth boundary points).

---

## Discretization

For simplicity, we discretize the boundary data using **piecewise
constant elements**, and evaluate the equation at the barycenter of
each boundary element.

---