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

def plot_all(R,r,n):
    pde = ws.PoissonSolver()
    
    I = 1.0 ; j = I / (pi * r**2)
    pde.set_nwalks(500_00)
    pde.set_sdfR(100.0)
    pde.set_samplevol(np.pi * r**2 * R)
    pde.set_solution_dimension(3)
    
    # Define query points grid-like
    lower_corner = [-1.0,-1.0,-1.0] # Lower corner grid delimiter
    upper_corner = [1.0,1.0,1.0]    # Upper corner grid delimiter
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    pde.set_query_points(query_points)
    
    # Source points and values/vectors
    source_points, source_tans = generate_spire_points(R,r,10_000)
    source_tans *= j * (4.0*pi * 10**-7)
    pde.set_source_points(source_points)
    pde.set_source_values(source_tans)
    
    pde.solve()
    
    ws.ps_init()
    pde.plot()
    ps.show()
    
def flux_query_points(R,n):
    query_points = np.zeros((n,3), dtype=np.float32)
    query_tans   = np.zeros((n,3), dtype=np.float32)
    for i in range(n):
        theta = 2.0 * np.pi * random()
        query_points[i][0] = R * np.cos(theta)
        query_points[i][1] = R * np.sin(theta)
        query_points[i][2] = 0.0
        query_tans[i][0] = - np.sin(theta)
        query_tans[i][1] =   np.cos(theta)
        query_tans[i][2] = 0.0
    return query_points, query_tans
    
def analytical_inductance(R,r):
    mu0 = 4.0 * np.pi * 10**-7
    # return mu0 * R * (np.log(8.0*R/r) - 2.0 + 0.5)
    return mu0 * R * (np.log(8.0*R/r) - 2.0 + 0.5)
    
    
def compute_inductance(R,r):
    pde = ws.PoissonSolver()
    
    I = 1.0 ; j = I / (pi * r**2)
    pde.set_nwalks(1_000_00)
    pde.set_sdfR(1000.0)
    pde.set_samplevol(np.pi * r**2 * R)
    pde.set_solution_dimension(3)

    n_query = 1000
    query_points, query_tans = flux_query_points(R,n_query)  
    pde.set_query_points(query_points)

    # Source points and values/vectors
    source_points, source_tans = generate_spire_points(R,r,10_000)
    source_tans *= j * (4.0*pi * 10**-7)
    pde.set_source_points(source_points)
    pde.set_source_values(source_tans)
    
    # Solve
    pde.solve()
    
    # Flux
    l = 2.0 * np.pi * R
    phi = 0.0
    for i in range(n_query):
        phi += np.dot(pde.A[i],query_tans[i])
    phi *= (l/n_query)
    print("Flux       : ", phi)
    
    an_ind = analytical_inductance(R,r)
    print("Analytical : ", an_ind)

if __name__ == '__main__':
    R = 10
    r = 0.01
    n = 32
    # plot_all(R,r,n)
    compute_inductance(R,r)