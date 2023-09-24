from context import walkonspheres as ws
import polyscope as ps

from math import pi
import polyscope as ps
import numpy as np
from random import random
import fileinput
from time import time

def remove_characters(filename):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('{', ''), end='')
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace('}', ''), end='')

def compute_all(r,curvelen,n):
    pde = ws.PoissonSolver()
    
    pde.set_nwalks(200_000)
    pde.set_sdfR(1000.0)
    pde.set_samplevol(pi * r**2 * curvelen)
    pde.set_solution_dimension(3)
    
    # Define query points grid-like
    h = 6.0
    lower_corner = [-h,-h,-h] # Lower corner grid delimiter
    upper_corner = [h,h,h]    # Upper corner grid delimiter
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    pde.set_query_points(query_points)
    
    # Source points and values/vectors
    source_points_file = "assets/solenoid_points.dat"
    source_values_file = "assets/solenoid_tans.dat"
    remove_characters(source_points_file) ; remove_characters(source_values_file)
    pde.set_source_points(ws.read_pts_from_file(source_points_file))
    source_values = ws.read_pts_from_file(source_values_file)
    source_values *= (4.0*pi * 10**-7) * 40.0 / (pi * r**2)
    pde.set_source_values(source_values)
    
    pde.solve()
    
    ws.ps_init()
    sliceplane = ps.add_scene_slice_plane()
    sliceplane.set_pose([0.0,10.0,0.0],[0.0,-1.0,0.0])
    pcloud = pde.plot()
    pde.plot_mesh("assets/solenoid.stl")
    start = time()
    dx = 2.0*h/(n-1) ; pde.plot_rot(pcloud, n,n,n,dx,dx,dx)
    print("Plot time : ", time()-start)
    ps.show()
    
    
def flux_query_points(R,n):
    query_points = np.zeros((n,3), dtype=np.float32)
    query_tans   = np.zeros((n,3), dtype=np.float32)
    for i in range(n):
        theta = 2.0 * np.pi * random()
        query_points[i][0] = R*np.cos(theta)
        query_points[i][1] = R*np.sin(theta)
        query_points[i][2] = 0.0
        query_tans[i][0] = -np.sin(theta)
        query_tans[i][1] = np.cos(theta)
        query_tans[i][2] = 0.0
    return query_points, query_tans

def analytical_inductance(N,r,l,R):
    mu0 = 4.0 * np.pi * 10**-7
    return mu0 * N**2 * np.pi * R**2 / l
        
def compute_inductance(R,r,curvelen,n_loops, height):
    pde = ws.PoissonSolver()

    pde.set_nwalks(200_000)
    pde.set_sdfR(1000.0)
    pde.set_samplevol(pi * r**2 * curvelen)
    pde.set_solution_dimension(3)
    
    # Flux query points
    n_query = 200
    query_points, query_tans = flux_query_points(R,n_query)
    pde.set_query_points(query_points)
    
    # Source points and values/vectors
    pde.set_source_points(ws.read_pts_from_file("assets/solenoid_points.dat"))
    source_values = ws.read_pts_from_file("assets/solenoid_tans.dat")
    source_values *= (4.0*pi * 10**-7) / (pi * r**2)
    pde.set_source_values(source_values)
    
    pde.solve()
    
    length = 2.0 * np.pi * R
    print(length)
    phi = 0.0
    for i in range(n_query):
        phi += np.dot(pde.A[i],query_tans[i])
    phi *= (length/n_query)
    print("Flux       : ", phi)
    
    ind = n_loops * phi
    print("Inductance : ", ind)
    
    an_ind = analytical_inductance(n_loops, r, height, R)
    print("Analytical : ", an_ind)
        
    
if __name__ == '__main__':
    R = 1.0
    r = 0.25
    curvelen = 63.0
    n = 31
    n_loops = 10
    alpha = 0.1
    height = 2.0 * np.pi * n_loops * alpha
    
    # compute_inductance(R,r,curvelen,n_loops,height)

    compute_all(r,curvelen,n)