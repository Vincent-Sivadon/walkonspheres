from context import walkonspheres as ws
import polyscope as ps

from math import pi
import polyscope as ps
import numpy as np
from random import random

if __name__ == '__main__':
    pde = ws.PoissonSolver()
    
    R = 10.0
    r = 1.0
    curvelen = 8774.0
    pde.set_nwalks(200_000)
    pde.set_sdfR(1000.0)
    pde.set_samplevol(pi * r**2 * curvelen)
    pde.set_solution_dimension(3)
    
    # Define query points grid-like
    lower_corner = [-60.0,-60.0,-60.0] # Lower corner grid delimiter
    upper_corner = [60.0,60.0,60.0]    # Upper corner grid delimiter
    n = 31   # Solution grid dim
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    
    # Source points and values/vectors
    pde.set_query_points(query_points)
    pde.set_source_points(ws.read_pts_from_file("assets/solenoid_points.dat"))
    source_vectors = ws.read_pts_from_file("assets/solenoid_tans.dat")
    source_vectors *= (4.0*pi * 10**-7) / (pi * r**2)
    pde.set_source_vectors(source_vectors)
    
    pde.solve()
    
    ws.ps_init()
    # sliceplane = ps.add_scene_slice_plane()
    # sliceplane.set_pose([0.0,10.0,0.0],[0.0,-1.0,0.0])
    pde.plot()
    pde.plot_rot(n,120.0/60)
    ps.show()