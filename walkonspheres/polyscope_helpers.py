import polyscope as ps
import numpy as np

def ps_init():
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    ps.set_navigation_style("free")
    ps.set_up_dir("z_up")
    
def ps_add_pcloud(pcloud_name, pts):
    return ps.register_point_cloud(pcloud_name,pts,point_render_mode='quad',radius=0.001)

def ps_add_3D3(pcloud,arr,arr_name):
    pcloud.add_vector_quantity(arr_name,arr, enabled=True, color=(0.471,0.110,0.890))
    
def ps_add_3D1(pcloud,arr,arr_name):
    pcloud.add_scalar_quantity(arr_name,arr, enabled=False)
    
def ps_add_mesh(mesh,name):
    ps.register_surface_mesh(name,
                             mesh.points, mesh.cells[0].data,
                             smooth_shade=True,
                             color=(0.594,0.594,0.594),
                             transparency=0.153)

def ps_add_sliceplace():
    sliceplane = ps.add_scene_slice_plane()
    sliceplane.set_pose([0.0,0.0,0.0],[0.0,1.0,0.0])