import warp as wp
from math import pi

@wp.func
def green3D(R: wp.float32, d: wp.float32):
    return (1.0/(4.0*pi)) * (R-d)/(d*R)

@wp.kernel
def _rasterize_vector_kernel(A: wp.array(dtype=wp.vec3f),
                        # Solution query points
                        query_pts: wp.array(dtype=wp.vec3f),
                        # Stored gathered walk informations
                        wx: wp.array(dtype=wp.float32),     # Center coords of the walks
                        wy: wp.array(dtype=wp.float32),     # ..
                        wz: wp.array(dtype=wp.float32),     # .
                        whx: wp.array(dtype=wp.float32),    # Value of the walk (vector)
                        why: wp.array(dtype=wp.float32),    # ..
                        whz: wp.array(dtype=wp.float32),    # .
                        wr: wp.array(dtype=wp.float32),     # Radius of the walk
                        nstarts: wp.float32,
                        nwalks: wp.int32):                  # Number of walks (size of w- arrays)
    i = wp.tid()
    
    # Field value to be filled by this thread
    # = solution at point pt 
    Ai = wp.vec3f(0.0)
    pt = query_pts[i]
    
    for wid in range(nwalks):
        fy = wp.vec3f(whx[wid],why[wid],whz[wid]) / float(nstarts)
        d = wp.length(wp.vec3f(pt[0],pt[1],pt[2]) - wp.vec3f(wx[wid],wy[wid],wz[wid]))
        if (d<wr[wid]):
            Ai += fy * green3D(wr[wid],d)
    A[i] = Ai

@wp.kernel
def _rasterize_scalar_kernel(A: wp.array(dtype=wp.float32),
                        # Solution query points
                        query_pts: wp.array(dtype=wp.vec3f),
                        # Stored gathered walk informations
                        wx: wp.array(dtype=wp.float32),     # Center coords of the walks
                        wy: wp.array(dtype=wp.float32),     # ..
                        wz: wp.array(dtype=wp.float32),     # .
                        wh: wp.array(dtype=wp.float32),     # Value of the walk (scalar)
                        wr: wp.array(dtype=wp.float32),     # Radius of the walk
                        nstarts: wp.float32,
                        nwalks: wp.int32):                  # Number of walks (size of w- arrays)
    i = wp.tid()
    
    # Field value to be filled by this thread
    # = solution at point pt 
    Ai = float(0.0)
    pt = query_pts[i]
    
    for wid in range(nwalks):
        fy = wh[wid] / float(nstarts)
        d = wp.length(wp.vec3f(pt[0],pt[1],pt[2]) - wp.vec3f(wx[wid],wy[wid],wz[wid]))
        if (d<wr[wid]):
            Ai += fy * green3D(wr[wid],d)
    A[i] = Ai

def _rasterize_vector(A_h, A_d,
                query_points_d,
                w_xx_d, w_yx_d, w_zx_d,
                w_fx_d, w_fy_d, w_fz_d,
                w_rs_d,
                nstarts):
    n = query_points_d.shape[0]
    nwalks = w_rs_d.shape[0]
    wp.launch(kernel=_rasterize_vector_kernel,
              dim=n,
              inputs=[A_d,
                      query_points_d,
                      w_xx_d, w_yx_d, w_zx_d,
                      w_fx_d, w_fy_d, w_fz_d,
                      w_rs_d,
                      nstarts,nwalks])
    wp.copy(dest=A_h, src=A_d)
    wp.synchronize()
    return A_h.numpy()

def _rasterize_scalar(A_h, A_d,
                query_points_d,
                w_xx_d, w_yx_d, w_zx_d,
                w_f_d,
                w_rs_d,
                nstarts):
    n = query_points_d.shape[0]
    nwalks = w_rs_d.shape[0]
    wp.launch(kernel=_rasterize_scalar_kernel,
              dim=n,
              inputs=[A_d,
                      query_points_d,
                      w_xx_d, w_yx_d, w_zx_d,
                      w_f_d,
                      w_rs_d,
                      nstarts,nwalks])
    wp.copy(dest=A_h, src=A_d)
    wp.synchronize()
    return A_h.numpy()