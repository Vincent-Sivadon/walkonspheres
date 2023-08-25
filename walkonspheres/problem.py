import numpy as np
import warp as wp
import polyscope as ps
from time import time

from .utils import _create_solution_array, _create_walksinfo_arrays, _arrays_to_gpu, _compute_rot
from .mesh_utils import _readmesh
from .rasterize import _rasterize_vector, _rasterize_scalar
from .polyscope_helpers import ps_init,ps_add_pcloud,ps_add_3D3,ps_add_3D1,ps_add_mesh,ps_add_sliceplace
from cppcore import gensourcewalks_vector, gensourcewalks_scalar        

class PoissonSolver():
    odim = 3            # Output dimension (1: scalar field, 3: vector field)
    sdfR = 400.0        # Size of the "infinite" domain
    nwalks = 1000       # Number of maximum walk information we will get
    
    def solve(self):
        if (not hasattr(self, 'query_points')):
            raise ValueError("Query points not set (use set_query_points)")
        if (not hasattr(self, 'source_points')):
            raise ValueError("Source points not set")
        wp.init()
        A_h, A_d = _create_solution_array(len(self.query_points), self.odim)
        w_coords,w_vals,w_radius = _create_walksinfo_arrays(self.nwalks, self.odim)
        
        start = time()
        if (self.odim == 1):
            nstarts = gensourcewalks_scalar(self.source_points,
                                            self.source_values,
                                            w_coords, w_vals, w_radius,
                                            self.sdfR, self.nwalks)
        elif (self.odim == 3):
            nstarts = gensourcewalks_vector(self.source_points,
                                            self.source_vectors,
                                            w_coords, w_vals, w_radius,
                                            self.sdfR, self.nwalks)
        end = time()
        print("Source walk generation : ", end-start)
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
        end = time()
        print("Rasterization : ", end-start)
            
    def _compute_solution_magnitude(self):
        if (self.odim == 1):
            return
        n = len(self.query_points)
        self.A_mag = np.zeros(n, dtype=np.float32)
        for i in range(n):
            self.A_mag[i] = np.linalg.norm(self.A[i])
            
    def plot(self):
        pcloud_name = "Solution query points"
        pcloud = ps_add_pcloud(pcloud_name,self.query_points,len(self.query_points))
        if (self.odim == 1):
            ps_add_3D1(pcloud, self.A, "Solution")
        elif (self.odim == 3):
            self._compute_solution_magnitude()
            ps_add_3D1(pcloud, self.A_mag, "Magnitude")
            ps_add_3D3(pcloud,self.A,"Solution")
        ps_add_sliceplace()
        self.pcloud = pcloud
    def plot_mesh(self, meshfile):
        mesh = _readmesh(meshfile)
        ps_add_mesh(mesh, "Source Mesh")
    def plot_rot(self, ngrid, dx):
        B = _compute_rot(self.A,ngrid,dx)
        print(B)
        ps_add_3D3(self.pcloud, B, "Rotational")
    
    # Set all parameters
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
    
    # Tools to set source value associated with source_points from constant value, array
    # Scalar :
    def set_source_value(self,value):
        if (self.odim != 1):
            raise ValueError("Solution dimension isn't 1 (scalar)")
        self.source_values = value * np.ones(len(self.source_points), dtype=np.float32)
    def set_source_values(self,source_values):
        if (self.odim != 1):
            raise ValueError("Solution dimension isn't 1 (scalar)")
        if (len(self.source_points) != len(source_values)):
            raise ValueError("Source points and source_values given aren't the same length")
        self.source_values = source_values
    # Vector :
    def set_source_vector(self,vector):
        if (self.odim != 3):
            raise ValueError("Solution dimension isn't 3 (vector)")
        self.source_vectors = vector * np.ones((len(self.source_points),3), dtype=np.float32)
    def set_source_vectors(self,source_vectors):
        if (self.odim != 3):
            raise ValueError("Solution dimension isn't 3 (vector)")
        if (len(self.source_points) != len(source_vectors)):
            raise ValueError("Source points and source_vectors given aren't the same length")
        self.source_vectors = source_vectors