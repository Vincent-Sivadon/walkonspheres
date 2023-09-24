import numpy as np
import warp as wp
import polyscope as ps
from time import time

from .utils import _create_solution_array, _create_walksinfo_arrays, _arrays_to_gpu, _compute_magnitude
from .mesh_utils import _readmesh
from .rasterize import _rasterize_vector, _rasterize_scalar
from .polyscope_helpers import ps_init,ps_add_pcloud,ps_add_3D3,ps_add_3D1,ps_add_mesh
from cppcore import gensourcewalks_vector, gensourcewalks_scalar, compute_rot

class PoissonSolver():
    """Class that carries tools to solve a Poisson's equation with null boundary conditions,
    on a sphere domain that should be close to infinite, inside which a structure carries the
    equation source.
    
    All of the attributes below, except for the solution A, must be defined to run simulations
    
    Attributes:
    -----------
        odim                Output dimension (1: scalar field, 3: vector field)
        sdfR                Size of the "infinite" sphere domain
        nwalks              Number of walk information
        query_points        Array of points where we are searching the solution
        source_points       Pre-randomly-sampled 3D points coming from the source structure  
        source_values       Source values corresponding to source_points
        samplevol           Structure volume
        A                   PDE's solution
    -----------
    """
    
    odim = 3
    sdfR = 400.0
    nwalks = 1000
    
    def solve(self):
        "Method to compute the solution A, when all parameters have been set."
        
        # Check if some haven't been set yet (be careful some attributes has default values)
        if (not hasattr(self, 'query_points')):
            raise ValueError("!!Query points not set (use set_query_points)!!")
        if (not hasattr(self, 'source_points')):
            raise ValueError("!!Source points not set!!")
        if (not hasattr(self, 'source_values')):
            raise ValueError("!!Source values not set!!")
        if (not hasattr(self, 'samplevol')):
            raise ValueError("!!Structure volume (samplevol) not set!!")
        
        # Warp (NVIDIA) initialization
        wp.init()
        
        # Host and device solution data structures
        A_h, A_d = _create_solution_array(len(self.query_points), self.odim)
        
        # Structures to store walks informations (CPU and GPU transformation)
        w_coords,w_vals,w_radius = _create_walksinfo_arrays(self.nwalks, self.odim)
        
        # Walks information generation (C++ functions)
        # ----------------------------
        start = time()
        if (self.odim == 1):
            nstarts = gensourcewalks_scalar(self.source_points,
                                            self.source_values,
                                            w_coords, w_vals, w_radius,
                                            self.sdfR, self.nwalks, self.samplevol)
        elif (self.odim == 3):
            nstarts = gensourcewalks_vector(self.source_points,
                                            self.source_values,
                                            w_coords, w_vals, w_radius,
                                            self.sdfR, self.nwalks, self.samplevol)
        print("Source walk generation : ", time()-start)
        
        # Rasterization
        # -------------
        start = time()
        w_coords_d, w_vals_d, w_radius_d, query_points_d =_arrays_to_gpu(w_coords,w_vals,w_radius,self.query_points,self.odim)
        if (self.odim == 1):
            self.A = _rasterize_scalar(A_h,A_d,
                    query_points_d,
                    w_coords_d[0],w_coords_d[1],w_coords_d[2],
                    w_vals_d,
                    w_radius_d,
                    nstarts)
        elif (self.odim == 3):
            self.A = _rasterize_vector(A_h,A_d,
                    query_points_d,
                    w_coords_d[0],w_coords_d[1],w_coords_d[2],
                    w_vals_d[0],w_vals_d[1],w_vals_d[2],
                    w_radius_d,
                    nstarts)
        print("Rasterization : ", time()-start)
            
            
    def plot(self):
        "Plot PDE's solution. Returns a Point Cloud structure that carries the informations"
        
        pcloud_name = "Solution query points"
        pcloud = ps_add_pcloud(pcloud_name,self.query_points)
        if (self.odim == 1):
            ps_add_3D1(pcloud, self.A, "Scalar Solution")
        elif (self.odim == 3):
            ps_add_3D1(pcloud, _compute_magnitude(self.A), "A (mag)")
            ps_add_3D3(pcloud,self.A,"A (magnetic vector potential)")
        return pcloud
    
    def plot_mesh(self, meshfile):
        "Add mesh (from STL `meshfile`) to the plot"
        
        mesh = _readmesh(meshfile)
        ps_add_mesh(mesh, "Source Mesh")
        
    def plot_rot(self, pcloud, nx, ny, nz, dx, dy, dz):
        "Add PDE's solution's rotationnal to the plot"

        B = compute_rot(self.A, nx,ny,nz, dx,dy,dz)
        B = np.array(B, dtype=np.float32)
        B_mag = _compute_magnitude(B)
        ps_add_3D3(pcloud, B, "B (magnetic field)")
        ps_add_3D1(pcloud, B_mag, "B (mag)")
    
    """
    -------------
    Set functions
    -------------
    """

    def set_query_points(self, query_points):
        self.query_points = query_points
    def set_source_points(self, source_points):
        self.source_points = source_points  # array of dim (N,3)
    def set_nwalks(self, nwalks):
        self.nwalks = nwalks
    def set_sdfR(self, sdfR):
        self.sdfR = sdfR
    def set_solution_dimension(self,odim):
        self.odim = odim
    def set_samplevol(self,vol):
        self.samplevol = vol
    def set_source_values(self,source_values):
        self.source_values = source_values