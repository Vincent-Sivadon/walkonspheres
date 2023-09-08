from context import walkonspheres as ws
import cppcore
import polyscope as ps

from math import pi,cos,sin,sqrt
import polyscope as ps
import numpy as np
from random import random 

# Define geometry (single spire)
def generate_spire_points(R,r,n):
    points = np.zeros((n,3), dtype=np.float32)
    tans   = np.zeros((n,3), dtype=np.float32)
    for i in range(n):
        rho = r * sqrt(random())
        theta = 2.0*np.pi*random()
        phi = 2.0*np.pi*random()
        points[i][0] = (R + rho*np.cos(theta)) * np.cos(phi)
        points[i][1] = (R + rho*np.cos(theta)) * np.sin(phi)
        points[i][2] = rho * np.sin(theta)
        tans[i][0] = - np.sin(phi)
        tans[i][1] = np.cos(phi)
        tans[i][2] = 0
    return points,tans


if __name__ == '__main__':
    pde = ws.PoissonSolver()
    
    R = 0.5 ; r = 0.01 ; I = 1.0 ; j = I / (pi * r**2)
    pde.set_nwalks(500_00)
    pde.set_sdfR(2.0)
    pde.set_samplevol(np.pi * r**2 * R)
    pde.set_solution_dimension(3)
    
    # Define query points grid-like
    lower_corner = [-1.0,-1.0,-1.0] # Lower corner grid delimiter
    upper_corner = [1.0,1.0,1.0]    # Upper corner grid delimiter
    n = 31   # Solution grid dim
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    
    # Source points and values/vectors
    pde.set_query_points(query_points)
    source_points, source_tans = generate_spire_points(R,r,10_000)
    source_tans *= j * (4.0*pi * 10**-7)
    pde.set_source_points(source_points)
    pde.set_source_vectors(source_tans)
    
    pde.solve()
    
    ws.ps_init()
    pde.plot()
    ps.show()