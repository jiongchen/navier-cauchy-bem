import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from loguru import logger
import argparse
from scipy.spatial import Delaunay

from somigliana.sample import generate_internal_points, generate_boundary_elements
from somigliana.bie_solver import BIESolver

jax.config.update("jax_enable_x64", True)

FLAG_PLOT = True
IMG_ROW = 1
IMG_COL = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command-line parser")

    parser.add_argument("--num_boundary", type=int)
    parser.add_argument("--num_eval", type=int)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--nu", type=float)
    
    args = parser.parse_args()
    print(args)
    
    logger.info(f"Default device: {jax.default_backend()}")
    logger.info(f"Devices available: {jax.devices()}")
    logger.info(f"GPU available: {jax.devices("gpu")}")

    # domain configuration
    shape_params = {
        'boundary_type': 'rectangle',
        'x_min': -1,
        'x_max': 1,
        'y_min': -1,
        'y_max': 1,
        'n_x': args.num_boundary // 4,
        'n_y': args.num_boundary // 4
    }

    mu = args.mu
    nu = args.nu

    # geometry: boundary elements and evaluation nodes
    bnd_nodes, bnd_elem = generate_boundary_elements(shape_params)
    eval_nodes = generate_internal_points(shape_params, args.num_eval)
    eval_trigs = Delaunay(eval_nodes)
    logger.critical(f"number of boundary elements={bnd_elem.shape[0]}")
    logger.critical(f"number of eval nodes={eval_nodes.shape[0]}")
    
    # solver for nonlinear elasticity
    solver = BIESolver(bnd_nodes, bnd_elem, mu, nu)

    # boundary condition
    boundary_condition = {
        'type': jnp.ones((solver.num_boundary_elements(), ), dtype=int), # 0-Dirichlet, 1-Neumann
        'value': jnp.zeros((solver.num_boundary_elements(), 2), dtype=float)
    }
    top_mask = jnp.abs(solver.centers[:, 1] - shape_params['y_max']) < 1e-8
    bottom_mask = jnp.abs(solver.centers[:, 1] - shape_params['y_min']) < 1e-8    
    boundary_condition['type'] = boundary_condition['type'].at[top_mask].set(0)
    boundary_condition['type'] = boundary_condition['type'].at[bottom_mask].set(0)
    boundary_condition['value'] = boundary_condition['value'].at[top_mask].set(jnp.array([0, 0.5]))
    boundary_condition['value'] = boundary_condition['value'].at[bottom_mask].set(jnp.array([0, -0.5]))

    # solve and interpolate
    solver.assemble_kernel_matrices()
    solver.solve(boundary_condition)
    eval_result = solver.evaluate_solution(eval_nodes)

    if FLAG_PLOT:
        plt.subplot(IMG_ROW, IMG_COL, 1)
        plt.scatter(solver.centers[:, 0], solver.centers[:, 1], color='blue', marker='*', s=5, label='centers')
        plt.scatter(bnd_nodes[:, 0], bnd_nodes[:, 1], color='gray', marker='o', s=50, label='boundary')

        for i in range(bnd_elem.shape[0]):
            n1, n2 = bnd_elem[i]
            x1, x2 = bnd_nodes[n1], bnd_nodes[n2]
            plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='violet', linestyle='-')

        for i in range(bnd_elem.shape[0]):
            center = solver.centers[i]
            normal = 0.2*solver.normals[i]
            plt.plot([center[0], center[0]+normal[0]], [center[1], center[1]+normal[1]], color='teal', linestyle='--')

        plt.title("boundary elements")
        plt.gca().set_aspect('equal')        

        # bc type
        plt.subplot(IMG_ROW, IMG_COL, 2)
        plt.scatter(solver.centers[:, 0], solver.centers[:, 1], c=boundary_condition['type'])
        plt.title("boundary condition types")
        plt.gca().set_aspect('equal')                
        
        # solution
        plt.subplot(IMG_ROW, IMG_COL, 3)
        plt.triplot(eval_nodes[:, 0]+eval_result[:, 0], eval_nodes[:, 1]+eval_result[:, 1], eval_trigs.simplices, color='k', linewidth=0.8)
        plt.title("deformation")
        plt.gca().set_aspect('equal')        

        plt.show()