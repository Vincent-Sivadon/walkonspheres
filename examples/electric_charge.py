from context import walkonspheres as ws
import cppcore

from math import pi
import polyscope as ps
import numpy as np
from random import random

def generate_sphere_points(R,n):
    points = np.zeros((n,3), dtype=np.float32)
    values  = np.zeros(n, dtype=np.float32)
    for i in range(n):
        theta = np.arccos(2.0*random() - 1.0)
        phi = 2.0*np.pi * random()
        rho = R * np.cbrt(random())
        points[i][0] = rho*np.sin(theta)*np.cos(phi)
        points[i][1] = rho*np.sin(theta)*np.sin(phi)
        points[i][2] = rho*np.cos(theta)
        values[i] = 1.0
    return points, values

if __name__ == '__main__':
    pde = ws.PoissonSolver()
    
    R = 1.0
    eps0 = 8.854 * 10**-12
    V = 4.0/3.0 * np.pi * R**3
    Q = 1.0
    rho = Q/V
    
    pde.set_nwalks(10000)
    pde.set_sdfR(500.0)
    pde.set_samplevol(V)
    pde.set_solution_dimension(1)
    
    # Define query points grid-like
    lower_corner = [-10.0,-10.0,-10.0] # Lower corner grid delimiter
    upper_corner = [10.0,10.0,10.0]    # Upper corner grid delimiter
    n = 50   # Solution grid dim
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    
    # Source points and values/vectors
    pde.set_query_points(query_points)
    points, values = generate_sphere_points(R,1000)
    pde.set_source_points(points)
    values *= rho/eps0
    pde.set_source_values(values)
    
    pde.solve()
    
    ws.ps_init()
    pde.plot()
    ps.show()