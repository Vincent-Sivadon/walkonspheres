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
    h = 9.0
    lower_corner = [-h,-h,-h] # Lower corner grid delimiter
    upper_corner = [h,h,h]    # Upper corner grid delimiter
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    pde.set_query_points(query_points)
    
    # Source points and values/vectors
    source_points_file = "assets/torus_points.dat"
    source_values_file = "assets/torus_tans.dat"
    remove_characters(source_points_file) ; remove_characters(source_values_file)
    pde.set_source_points(ws.read_pts_from_file(source_points_file))
    source_values = ws.read_pts_from_file(source_values_file)
    source_values *= (4.0*pi * 10**-7) * 400.0 / (pi * r**2)
    pde.set_source_values(source_values)
    
    pde.solve()
    
    start = time()
    ws.ps_init()
    sliceplane = ws.ps_add_sliceplane()
    pcloud = pde.plot()
    pde.plot_mesh("assets/torus.stl")
    dx = 2.0*h/(n-1) ; pde.plot_rot(pcloud, n,n,n,dx,dx,dx)
    print("Plot time : ", time()-start)
    ps.show()
    
    

if __name__ == '__main__':
    r = 0.1
    curvelen = 0.031
    n = 31
    compute_all(r,curvelen,n)
    
    
    
