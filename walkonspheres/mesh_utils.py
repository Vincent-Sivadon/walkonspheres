import meshio as ms
import warp as wp

def _readmesh(meshfile):
    return ms.read(meshfile)
   
# Read mesh parameters from file
# File should always start with the mesh surface area
def _read_mesh_param_file(mesh_param_file):
    file = open(mesh_param_file, 'r')
    lines = file.readlines()
    mesh_params = []
    for l in lines:
        mesh_params.append(float(l))
    return mesh_params