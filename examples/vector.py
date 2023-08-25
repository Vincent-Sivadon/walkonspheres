from context import walkonspheres as ws
import cppcore
import polyscope as ps

from math import pi
import polyscope as ps

if __name__ == '__main__':
    pde = ws.PoissonSolver()
    
    pde.set_nwalks(500_000)
    pde.set_sdfR(400.0)
    pde.set_solution_dimension(3)
    
    # Define query points grid-like
    lower_corner = [-170.0,-170.0,-170.0] # Lower corner grid delimiter
    upper_corner = [170.0,170.0,170.0]    # Upper corner grid delimiter
    n = 30   # Solution grid dim
    query_points = ws.create_grid_query_pts(lower_corner, upper_corner, n)
    
    # Source points and values/vectors
    pde.set_query_points(query_points)
    pde.set_source_points(ws.read_pts_from_file("assets/source_points.dat"))
    pde.set_source_vectors(ws.read_pts_from_file("assets/source_vectors.dat"))
    
    pde.solve()
    
    ws.ps_init()
    sliceplane = ps.add_scene_slice_plane()
    sliceplane.set_pose([0.0,10.0,0.0],[0.0,-1.0,0.0])
    pde.plot()
    pde.plot_mesh("assets/mesh.stl")
    pde.plot_rot(n,340.0/29)
    ps.show()