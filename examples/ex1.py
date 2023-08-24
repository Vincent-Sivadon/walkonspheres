from context import walkonspheres as ws
import cppcore

from math import pi
import polyscope as ps

if __name__ == '__main__':
    pde = ws.PoissonSolver()
    
    pde.set_nwalks(10000)
    pde.set_sdfR(400.0)
    pde.set_solution_dimension(1)
    
    # Define query points grid-like
    lower_corner = [-170.0,-170.0,-170.0] # Lower corner grid delimiter
    upper_corner = [170.0,170.0,170.0]    # Upper corner grid delimiter
    n = 30   # Solution grid dim
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    
    # Source points and values/vectors
    pde.set_query_points(query_points)
    pde.set_source_points(ws.read_pts_from_file("assets/meshpts.dat"))
    pde.set_source_value(1.0)
    # pde.set_source_vectors(ws.read_pts_from_file("assets/meshpts_tangents.dat"))
    
    pde.solve()
    
    pde.plot_with_mesh("assets/mesh.stl")