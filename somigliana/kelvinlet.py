import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial
from scipy.special import roots_legendre

jax.config.update("jax_enable_x64", True)

def gauss_quadrature_0_1(n):
    nodes, weights = roots_legendre(n)
    
    # Map nodes from [-1, 1] to [0, 1]
    nodes = 0.5 * (nodes + 1)
    weights = 0.5 * weights  # Scale weights accordingly

    return nodes, weights


DENSE_GAUSS_S, DENSE_GAUSS_W = gauss_quadrature_0_1(256)


@jax.jit
def kelvin_solution(x, y, mu, nu):
    """
    2D Kelvin solution
    """
    a = 1.0/(2*jnp.pi*mu)
    b = a/(4-4*nu)
    r = x - y
    r2 = jnp.dot(r, r)
    return -(a-b)/2.0*jnp.log(r2)*jnp.eye(2) + b/r2 * jnp.outer(r, r)


@jax.jit
def kelvin_traction(x, y, n, mu, nu):
    """
    2D Kelvin traction
    """
    a = 1.0/(2*jnp.pi*mu)
    b = a/(4-4*nu)
    r = x - y
    r2 = jnp.dot(r, r)
    return mu*(a-2*b)/r2*(jnp.dot(n, r)*jnp.eye(2) + jnp.outer(n, r) - jnp.outer(r, n)) + 4*mu*b/r2**2 * jnp.dot(n, r) * jnp.outer(r, r)


@jax.jit
def integrate_K_over_element(x, p, q, mu, nu, gs, gw):
    mid = 0.5*(p+q)
    area = jnp.linalg.norm(p-q)

    # check singularity
    r = x-mid
    rnorm = jnp.linalg.norm(r)

    # kelvinlet is weakly singular, integral exists,
    # so it's safe to use many quadratures
    def true_fn():
        def one_entry_K(s, w):
            y = (1-s)*p + s*q
            return w*kelvin_solution(x, y, mu, nu)

        Gs = jax.vmap(one_entry_K, in_axes=(0, 0))(DENSE_GAUSS_S, DENSE_GAUSS_W)
        return jnp.sum(Gs, axis=0)*area

    def false_fn():
        G = jnp.zeros((2, 2))
        for s, w in zip(gs, gw):
            y = (1-s)*p + s*q
            G += w*kelvin_solution(x, y, mu, nu)
        return G*area

    return jax.lax.cond(rnorm < 1e-8, true_fn, false_fn)


@jax.jit
def integrate_T_over_element(x, p, q, n, mu, nu, gs, gw):
    mid = 0.5*(p+q)
    area = jnp.linalg.norm(p-q)

    # check singularity
    r = x-mid
    rnorm = jnp.linalg.norm(r)

    # Cauchy principle value is zero
    def true_fn():
        return 0.5*jnp.eye(2)

    def false_fn():
        G = jnp.zeros((2, 2))
        for s, w in zip(gs, gw):
            y = (1-s)*p + s*q
            G += w*kelvin_traction(x, y, n, mu, nu)
        return G*area

    return jax.lax.cond(rnorm < 1e-8, true_fn, false_fn)

    
if __name__ == "__main__":
    key = random.PRNGKey(42)
    xy = random.uniform(key, shape=(2, 2))
    x, y = xy[0], xy[1]
    n = jnp.ones((2,))
    n /= jnp.linalg.norm(n)
    print(x.shape, y.shape, n.shape)

    mu = 1.0
    nu = 0.4

    @jax.jit
    def disp(x, y, w, mu, nu):
        return kelvin_solution(x, y, mu, nu) @ w

    @jax.jit
    def stress(x, y, w, mu, nu):
        J = jax.jacobian(disp, argnums=1)(x, y, w, mu, nu)
        return mu*(J + J.T) + 2*mu*nu/(1-2*nu)*jnp.trace(J)*jnp.eye(2)
    
    print(kelvin_solution(x, y, mu, nu))
    print(kelvin_traction(x, y, n, mu, nu))

    w = jnp.ones((2, ))
    J = jax.jacobian(stress, argnums=2)(x, y, w, mu, nu)

    res = jnp.tensordot(J, n, axes=([1], [0])).T
    print(res)