import numpy as np
import warp as wp

# Returns an array that contains axis representing a grid of n*n*n points
# Example : lower_corner = [-1.0,-1.0,-1.0], upper_corner = [1.0,1.0,1.0], n=4 :
#   axs[0] = [-1,-0.5,0.5,1.0] (X axis)
#   axs[1] = [-1,-0.5,0.5,1.0] (Y axis)
#   axs[2] = [-1,-0.5,0.5,1.0] (Z axis)
def _create_grid_axs(lower_corner, upper_corner, n):
    axs = []
    for i in range(3):
        axs.append(np.linspace(lower_corner[i], upper_corner[i], n))
    return axs

def create_grid_query_pts(lower_corner, upper_corner, n):
    axs = _create_grid_axs(lower_corner, upper_corner, n)
    query_pts = np.zeros((n*n*n,3), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                query_pts[k*n*n + i*n + j][0] = axs[0][i]
                query_pts[k*n*n + i*n + j][1] = axs[1][j]
                query_pts[k*n*n + i*n + j][2] = axs[2][k]
    return query_pts
                
def _create_vector_arrays(n):
    arr_h = wp.empty(shape=n, dtype=wp.vec3f, device="cpu")
    arr_d = wp.empty(shape=n, dtype=wp.vec3f, device="cuda")
    return arr_h, arr_d

def read_pts_from_file(pts_file):
    pts = np.loadtxt(pts_file,delimiter=',')
    return pts.astype(np.float32)

def _create_solution_array(n,odim):
    if (odim == 1):
        arr_h = wp.empty(shape=n, dtype=wp.float32, device="cpu")
        arr_d = wp.empty(shape=n, dtype=wp.float32, device="cuda")
    elif (odim == 3):
        arr_h = wp.empty(shape=n, dtype=wp.vec3f, device="cpu")
        arr_d = wp.empty(shape=n, dtype=wp.vec3f, device="cuda")
    else:
        raise ValueError("Solution dimension set to a number different than 1 or 3")
    return arr_h, arr_d
    
def _create_walksinfo_arrays(nwalks, odim):
        w_coords = [np.zeros(nwalks, dtype=np.float32) for _ in range(3)]
        w_coords = np.array(w_coords)
        # Walk values
        if (odim==1):
            w_vals = np.zeros(nwalks, dtype=np.float32)
        elif (odim==3):
            w_vals = []
            for _ in range(odim):
                w_vals.append(np.zeros(nwalks, dtype=np.float32))
            w_vals = np.array(w_vals) ; w_vals.reshape(odim,nwalks)
        else:
            raise ValueError("Solution dimension different than 1 or 3")          
        # Walk radius
        w_radius = np.zeros(nwalks, dtype=np.float32)
        
        return w_coords, w_vals, w_radius
    
def _arrays_to_gpu(w_coords,w_vals,w_radius,query_points,odim):
    w_coords_d = []
    for i in range(3):
        w_coords_d.append(wp.from_numpy(w_coords[i], dtype=wp.float32, device="cuda"))
    if (odim==1):
        w_vals_d = wp.from_numpy(w_vals, dtype=wp.float32, device="cuda")
    elif (odim==3):
        w_vals_d = []
        for i in range(3):
            w_vals_d.append(wp.from_numpy(w_vals[i], dtype=wp.float32, device="cuda"))
    else:
        raise ValueError("Solution dimension different than 1 or 3") 
    w_radius_d = wp.from_numpy(w_radius, dtype=wp.float32, device="cuda")
    query_points_d = wp.from_numpy(query_points, dtype=wp.vec3f, device="cuda")
    
    return w_coords_d, w_vals_d, w_radius_d, query_points_d