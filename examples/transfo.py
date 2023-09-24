from context import walkonspheres as ws
import polyscope as ps

from math import pi
import polyscope as ps
from time import time
from random import randint
import fileinput
import numpy as np

def remove_characters(filename):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('{', ''), end='')
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('}', ''), end='')
    

def generate_median_plan_points(n):
    ax_x = np.linspace(-150.0,150.0, n)
    ax_y = np.linspace(-150.0,150.0, n)
    nz = 3
    ax_z = np.linspace(-1.0,1.0, nz)
    query_pts = np.zeros((n*n*nz,3), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(nz):
                query_pts[k*nz*n + i*n + j][0] = ax_x[i]
                query_pts[k*nz*n + i*n + j][1] = ax_y[j]
                query_pts[k*nz*n + i*n + j][2] = ax_z[k]
    return query_pts
    
def compute_all(r, vol, nx,ny,nz, Lx,Ly,Lz, query_points):
    n = nx
    pde = ws.PoissonSolver()
    
    pde.set_nwalks(2_000_000)
    pde.set_sdfR(2000.0)
    pde.set_samplevol(vol)
    pde.set_solution_dimension(3)
    
    pde.set_query_points(query_points)

    # Source points
    remove_characters("assets/transfo_tans.txt")
    remove_characters("assets/transfo_points.txt")
    pde.set_source_points(ws.read_pts_from_file("assets/transfo_points.txt"))
    source_values = ws.read_pts_from_file("assets/transfo_tans.txt")
    source_values *= (4.0*pi * 10**-7) * 40.0 / (pi * r**2)
    pde.set_source_values(source_values)
    
    pde.solve()
    
    ws.ps_init()
    # sliceplane = ps.add_scene_slice_plane()
    # sliceplane.set_pose([0.0,10.0,0.0],[0.0,-1.0,0.0])
    pcloud = pde.plot()
    pde.plot_mesh("assets/mesh.stl")
    pde.plot_rot(pcloud, nx,ny,nz,Lx/(nx-1),Ly/(ny-1),Lz/(nz-1))
    ps.show()
    
def compute_inductance(r, vol, int_curv_len):
    pde = ws.PoissonSolver()
    
    pde.set_nwalks(1_000_000)
    pde.set_sdfR(2000.0)
    pde.set_samplevol(vol)
    pde.set_solution_dimension(3)
    
    # File treatment
    start = time()
    query_points_file = "assets/transfo_flux_query_points.dat"
    query_tans_file   = "assets/transfo_flux_query_tans.dat"
    remove_characters(query_points_file)
    remove_characters(query_tans_file)
    remove_characters("assets/transfo_tans.txt")
    remove_characters("assets/transfo_points.txt")
    print("File gestion : ", time() - start)
    
    # Set query points
    query_points = ws.read_pts_from_file(query_points_file)
    query_tans   = ws.read_pts_from_file(query_tans_file)
    pde.set_query_points(query_points)

    # Source points/tans
    source_points = ws.read_pts_from_file("assets/transfo_points.txt")
    source_tans   = ws.read_pts_from_file("assets/transfo_tans.txt")
    source_tans *= (4.0*pi * 10**-7) / (pi * r**2)
    pde.set_source_points(source_points)
    pde.set_source_values(source_tans)
    
    # Solve
    pde.solve()
    
    # Compute flux
    phi = 0.0
    n_query = 1000
    n_max = len(query_points)
    for _ in range(n_query):
        i = randint(0,n_max-1)
        phi += np.dot(pde.A[i], query_tans[i])
    phi *= (int_curv_len/n_query)
    print("Flux       : ", phi)
    
    n_loops = 6.0 # serial
    n_elements = 10.0 # parallel
    ind = n_loops * phi / n_elements
    print("Inductance : ", ind)


if __name__ == '__main__':
    r = 0.002
    vol = 0.457764
    int_curv_len = 0.647

    # Median    
    # n = 300
    # query_points = generate_median_plan_points(n)
    # compute_all(r,vol,n,n,3,0.3,0.3,0.002,query_points)
    
    # Grid
    # n = 61 ; L = 150.0
    # lower_corner = [-L,-L,-L] # Lower corner grid delimiter
    # upper_corner = [L,L,L]    # Upper corner grid delimiter
    # query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    # compute_all(r,vol,n,n,n,2.0*L,2.0*L,2.0*L,query_points)
    
    compute_inductance(r,vol,int_curv_len)