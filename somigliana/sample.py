import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import triangle as tr


jax.config.update("jax_enable_x64", True)


def _create_circular_boundary(center, radius, n_elements):
    angles = jnp.linspace(0, 2*jnp.pi, n_elements + 1)[:-1]

    nodes = jnp.zeros((n_elements, 2))
    nodes = nodes.at[:, 0].set(center[0] + radius * jnp.cos(angles))
    nodes = nodes.at[:, 1].set(center[1] + radius * jnp.sin(angles))
    
    elements = jnp.zeros((n_elements, 2), dtype=int)
    for i in range(n_elements):
        elements = elements.at[i].set([i, (i + 1) % n_elements])
    
    return nodes, elements


def _create_rectangular_boundary(x_min, x_max, y_min, y_max, n_x, n_y):
    nodes = []
    
    # Bottom edge
    x_bottom = jnp.linspace(x_min, x_max, n_x + 1)
    for x in x_bottom[:-1]:  # Exclude last point to avoid duplication
        nodes.append([x, y_min])
    
    # Right edge
    y_right = jnp.linspace(y_min, y_max, n_y + 1)
    for y in y_right[:-1]:
        nodes.append([x_max, y])
    
    # Top edge (reverse order)
    x_top = jnp.linspace(x_max, x_min, n_x + 1)
    for x in x_top[:-1]:
        nodes.append([x, y_max])
    
    # Left edge (reverse order)
    y_left = jnp.linspace(y_max, y_min, n_y + 1)
    for y in y_left[:-1]:
        nodes.append([x_min, y])
    
    nodes = jnp.array(nodes)
    n_nodes = len(nodes)
    
    # Create elements
    elements = []
    for i in range(n_nodes):
        elements.append([i, (i + 1) % n_nodes])
    
    elements = jnp.array(elements)
    
    return nodes, elements


def _create_star_elements(n_tips, R0, R1, n_nodes):
    n_points = n_tips   # number of star tips
    theta = jnp.linspace(0, 2*jnp.pi, n_nodes + 1)[: -1]

    # Polar radius
    r = R0 + R1 * jnp.cos(n_points * theta)
    
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    nodes = jnp.column_stack([x, y])

    elements = []
    for i in range(n_nodes):
        elements.append([i, (i + 1) % n_nodes])
    elements = jnp.array(elements)  

    return nodes, elements
    

def generate_boundary_elements(domain_params):
    boundary_type = domain_params['boundary_type']
    
    if boundary_type == 'circle':
        center     = domain_params['center']
        radius     = domain_params['radius']
        n_elements = domain_params['n_elements']
        return _create_circular_boundary(center, radius, n_elements)

    elif boundary_type == 'rectangle':
        x_min = domain_params['x_min']
        x_max = domain_params['x_max']
        y_min = domain_params['y_min']
        y_max = domain_params['y_max']
        n_x   = domain_params['n_x']
        n_y   = domain_params['n_y']
        return _create_rectangular_boundary(x_min, x_max, y_min, y_max, n_x, n_y)

    elif boundary_type == 'star':
        num_tips = domain_params['n_tips']
        num_nodes = domain_params['n_nodes']
        R0 = domain_params['R0']
        R1 = domain_params['R1']
        return _create_star_elements(num_tips, R0, R1, num_nodes)

    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")   


def meshing_polygon(elem, bnd_nodes, area):
    # meshing
    poly = dict(vertices=np.array(bnd_nodes), segments=np.array(elem))
    mesh = tr.triangulate(poly, f'pq30a{area}')
    vertices = mesh['vertices']
    segments = mesh.get('segments', np.array([]))
    assert segments.size > 0

    boundary_indices = np.unique(segments)
    interior_indices = np.setdiff1d(np.arange(len(vertices)), boundary_indices)
    interior_nodes = vertices[interior_indices]

    return jnp.array(interior_nodes)

    
def generate_internal_points(domain_params=None, n_internal=20):
    boundary_type = domain_params['boundary_type']
    
    if boundary_type == 'circle':
        center = domain_params['center']
        radius = domain_params['radius']
        
        radii = jnp.linspace(0.1, 0.9, int(jnp.sqrt(n_internal)))
        points = []
        for r in radii:
            n_circle = max(1, int(n_internal * r / radii[-1]))
            theta = jnp.linspace(0, 2*jnp.pi, n_circle + 1)[:-1]
            for t in theta:
                x = center[0] + r * radius * jnp.cos(t)
                y = center[1] + r * radius * jnp.sin(t)
                points.append([x, y])
            
        return jnp.array(points)
                
    elif boundary_type == 'rectangle':
        x_min = domain_params['x_min']
        x_max = domain_params['x_max']  
        y_min = domain_params['y_min']
        y_max = domain_params['y_max']
        
        if True:
            # Uniform grid inside rectangle
            n_x = int(jnp.sqrt(n_internal))
            n_y = int(n_internal / n_x)
            
            x_int = jnp.linspace(x_min + 0.01*(x_max-x_min), x_max - 0.01*(x_max-x_min), n_x)
            y_int = jnp.linspace(y_min + 0.01*(y_max-y_min), y_max - 0.01*(y_max-y_min), n_y)

            X, Y = jnp.meshgrid(x_int, y_int)
            return jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    elif boundary_type == 'star':
        # Star parameters
        n_points = domain_params['n_tips']
        R0 = domain_params['R0']
        R1 = domain_params['R1']

        Ntheta = 50
        Nr = n_internal // Ntheta + 1
        
        def r_star(theta):
            return R0 + R1 * jnp.cos(n_points * theta)

        # Create grid
        theta = jnp.linspace(0, 2*jnp.pi, Ntheta, endpoint=False)
        r_norm = jnp.linspace(0.1, 0.9, Nr)

        # Use meshgrid
        Theta, Rnorm = jnp.meshgrid(theta, r_norm)
        R = Rnorm * r_star(Theta)  # scale by local boundary radius

        # Convert to Cartesian
        x = R * jnp.cos(Theta)
        y = R * jnp.sin(Theta)
        
        return jnp.column_stack([x.ravel(), y.ravel()])


if __name__ == '__main__':
    circle_params = {'boundary_type': 'circle', 'center': jnp.zeros((2,)), 'radius': 1.0, 'n_elements': 40}
    rectangle_params = {'boundary_type': 'rectangle', 'x_min': -1, 'x_max': 1, 'y_min': -1, 'y_max': 1, 'n_x': 20, 'n_y': 20}
    star_params = {'boundary_type': 'star', 'n_tips': 5, 'n_nodes': 100, 'R0': 1.0, 'R1': 0.3}

    bnd_nodes, elem = generate_boundary_elements(circle_params)
    int_nodes = generate_internal_points(circle_params, 36) 
    plt.subplot(2, 2, 1)
    plt.title(f'nb={bnd_nodes.shape[0]}, ni={int_nodes.shape[0]}')
    plt.scatter(bnd_nodes[:, 0], bnd_nodes[:, 1], c='b')
    plt.scatter(int_nodes[:, 0], int_nodes[:, 1], c='g')
    
    bnd_nodes, elem = generate_boundary_elements(rectangle_params)
    int_nodes = generate_internal_points(rectangle_params, 100)
    plt.subplot(2, 2, 2)
    plt.title(f'nb={bnd_nodes.shape[0]}, ni={int_nodes.shape[0]}')
    plt.scatter(bnd_nodes[:, 0], bnd_nodes[:, 1], c='b')
    plt.scatter(int_nodes[:, 0], int_nodes[:, 1], c='g')

    bnd_nodes, elem = generate_boundary_elements(star_params)
    int_nodes = generate_internal_points(star_params, 256)
    plt.subplot(2, 2, 3)
    plt.title(f'nb={bnd_nodes.shape[0]}, ni={int_nodes.shape[0]}')
    plt.scatter(bnd_nodes[:, 0], bnd_nodes[:, 1], c='b')
    plt.scatter(int_nodes[:, 0], int_nodes[:, 1], c='g')

    # # meshing
    # poly = dict(vertices=np.array(bnd_nodes), segments=np.array(elem))
    # mesh = tr.triangulate(poly, 'pq30a0.02')
    # vertices = mesh['vertices']
    # segments = mesh.get('segments', np.array([]))
    # if segments.size > 0:
    #     boundary_indices = np.unique(segments)
    #     interior_indices = np.setdiff1d(np.arange(len(vertices)), boundary_indices)
    # interior_nodes = vertices[interior_indices]
    
    # plt.subplot(2, 2, 4)
    # plt.triplot(vertices[:, 0], vertices[:, 1], mesh['triangles'])
    # plt.scatter(bnd_nodes[:,0], bnd_nodes[:,1], color='red')
    # plt.scatter(interior_nodes[:,0], interior_nodes[:, 1], color='gray', s=10)
    # plt.gca().set_aspect('equal')
    
    plt.show()    
