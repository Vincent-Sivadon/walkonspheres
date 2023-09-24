#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>
#include <random>

#define PI 3.14159265

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int gensourcewalks_vector(py::array_t<float> source_points,
                          py::array_t<float> source_vectors,
                          py::array_t<float> w_coords,
                          py::array_t<float> w_vals,
                          py::array_t<float> w_radius,
                          float sdfR, int nwalks, float samplevol)
{
        // Array accessors (c++-python interface specificity)
        auto a_source_points  = source_points.unchecked<2>();
        auto a_w_coords       = w_coords.mutable_unchecked<2>();
        auto a_w_vals         = w_vals.mutable_unchecked<2>();
        auto a_w_radius       = w_radius.mutable_unchecked<1>();
        auto a_source_vectors = source_vectors.unchecked<2>();

        // Total number of source_points
        py::ssize_t n_source_points = a_source_points.shape(0);

        // Initialize random seed needed to sample points from the mesh's surface
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n_source_points);

        int i = 0;
        int nstarts = 0;

        while (i < nwalks) {
            // Index of the element from source_points (will define the starting pt for the walk)
            int start_index = dist(gen);    

            float x = a_source_points(start_index,0);
            float y = a_source_points(start_index,1);
            float z = a_source_points(start_index,2);

            // Source term to be propagated
            float fx = a_source_vectors(start_index,0);
            float fy = a_source_vectors(start_index,1);
            float fz = a_source_vectors(start_index,2);

            // SDF : distance to boundaries (sphere of radius sdfR)
            float ry = sdfR - sqrtf(x*x + y*y + z*z);

            // Continue the walk until we reach the boundaries or we read max info stored
            while (0.01 < ry && i < nwalks) {
                // Store current walk information
                a_w_coords(0,i) = x;   // Position (sphere center)
                a_w_coords(1,i) = y;   // ...
                a_w_coords(2,i) = z;   // ...
                a_w_vals(0,i) = samplevol*fx; // Source term
                a_w_vals(1,i) = samplevol*fy; // Source term
                a_w_vals(2,i) = samplevol*fz; // Source term
                a_w_radius(i) = ry;           // Sphere radius

                // Select random numbers for the following
                float r1 = (float)rand() / (float)RAND_MAX;
                float r2 = (float)rand() / (float)RAND_MAX;

                // Sample a new point on the sphere of center {x,y,z} and radius ry
                float theta = acosf(2.0f * r1 - 1.0f);
                float phi = 2.0f * PI * r2;
                x += ry * sinf(theta) * cosf(phi);
                y += ry * sinf(theta) * sinf(phi);
                z += ry * cosf(theta);
                ry = sdfR - sqrtf(x*x + y*y + z*z);

                i++; // +1 information stored
            }
            nstarts++; // +1 walk compeleted
        }
        std::cout << "N walks started :" << nstarts << std::endl;
        return nstarts;
}

int gensourcewalks_scalar(py::array_t<float> source_points,
                          py::array_t<float> source_values,
                          py::array_t<float> w_coords,
                          py::array_t<float> w_vals,
                          py::array_t<float> w_radius,
                          float sdfR, int nwalks, float samplevol)
{
        // Array accessors (c++-python interface specificity)
        auto a_source_points  = source_points.unchecked<2>();
        auto a_w_coords       = w_coords.mutable_unchecked<2>();
        auto a_w_vals         = w_vals.mutable_unchecked<1>();
        auto a_w_radius       = w_radius.mutable_unchecked<1>();
        auto a_source_values  = source_values.unchecked<1>();

        // Total number of source_points
        py::ssize_t n_source_points = a_source_points.shape(0);

        // Initialize random seed needed to sample points from the mesh's surface
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n_source_points);

        int i = 0;
        int nstarts = 0;

        while (i < nwalks) {
            // Index of the element from source_points (will define the starting pt for the walk)
            int start_index = dist(gen);    
            float x = a_source_points(start_index,0);
            float y = a_source_points(start_index,1);
            float z = a_source_points(start_index,2);

            // Source term to be propagated
            float fy = a_source_values(start_index);

            // SDF : distance to boundaries (sphere of radius sdfR)
            float ry = sdfR - sqrtf(x*x + y*y + z*z);

            // Continue the walk until we reach the boundaries or we read max info stored
            while (0.01 < ry && i < nwalks) {
                // Store current walk information
                a_w_coords(0,i) = x;   // Position (sphere center)
                a_w_coords(1,i) = y;   // ...
                a_w_coords(2,i) = z;   // ...
                a_w_vals(i) = samplevol*fy;    // Source term
                a_w_radius(i) = ry;    // Sphere radius

                // Select random numbers for the following
                float r1 = (float)rand() / (float)RAND_MAX;
                float r2 = (float)rand() / (float)RAND_MAX;

                // Sample a new point on the sphere of center {x,y,z} and radius ry
                float theta = acosf(2.0f * r1 - 1.0f);
                float phi = 2.0f * PI * r2;
                x += ry * sinf(theta) * cosf(phi);
                y += ry * sinf(theta) * sinf(phi);
                z += ry * cosf(theta);
                ry = sdfR - sqrtf(x*x + y*y + z*z);

                i++; // +1 information stored
            }
            nstarts++; // +1 walk compeleted
        }
        std::cout << "N walks started :" << nstarts << std::endl;
        return nstarts;
}

std::vector<std::vector<float>> compute_rot(py::array_t<float> A, int nx, int ny, int nz, float dx, float dy, float dz)
{
    std::vector<std::vector<float>> B(nx*ny*nz, std::vector<float>(3));
    auto acc_A  = A.unchecked<2>();

    for (int i=1 ; i<nx-1 ; i++)
        for (int j=1 ; j<ny-1 ; j++)
            for (int k=1 ; k<nz-1 ; k++) {
                float dyAx = (acc_A(k*nz*nx + i*nx + j+1,0) - acc_A(k*nz*nx + i*nx + j-1,0))/(2.0*dy);
                float dzAx = (acc_A((k+1)*nz*nx + i*nx + j,0) - acc_A((k-1)*nz*nx + i*nx + j,0))/(2.0*dz);
                float dzAy = (acc_A((k+1)*nz*nx + i*nx + j,1) - acc_A((k-1)*nz*nx + i*nx + j,1))/(2.0*dz);
                float dxAy = (acc_A(k*nz*nx + (i+1)*nx + j,1) - acc_A(k*nz*nx + (i-1)*nx + j,1))/(2.0*dx);
                float dyAz = (acc_A(k*nz*nx + i*nx + (j+1),2) - acc_A(k*nz*nx + i*nx + (j-1),2))/(2.0*dy);
                float dxAz = (acc_A(k*nz*nx + (i+1)*nx + j,2) - acc_A(k*nz*nx + (i-1)*nx + j,2))/(2.0*dx);
                B[k*nz*nx + i*nx + j][0] = (dyAz - dzAy);
                B[k*nz*nx + i*nx + j][1] = (dzAx - dxAz);
                B[k*nz*nx + i*nx + j][2] = (dxAy - dyAx);
                // float Bx = (dyAz - dzAy) ; float By = (dzAx - dxAz) ; float Bz = (dxAy - dyAx);
                // float Bnorm = sqrtf(Bx*Bx + By*By * Bz*Bz);
                // B[k*nz*nx + i*nx + j][0] *= logf(Bnorm);
                // B[k*nz*nx + i*nx + j][1] *= logf(Bnorm);
                // B[k*nz*nx + i*nx + j][2] *= logf(Bnorm);
            }

    return B;
}

PYBIND11_MODULE(cppcore, m) {
    m.def(
        "gensourcewalks_vector", &gensourcewalks_vector,
        py::arg("source_points").noconvert(),
        py::arg("source_vectors").noconvert(),
        py::arg("w_coords").noconvert(),
        py::arg("w_vals").noconvert(),
        py::arg("w_radius").noconvert(),
        py::arg("sdfR").noconvert(),py::arg("nwalks").noconvert(),py::arg("samplevol").noconvert()
    );
    m.def(
        "gensourcewalks_scalar", &gensourcewalks_scalar,
        py::arg("source_points").noconvert(),
        py::arg("source_values").noconvert(),
        py::arg("w_coords").noconvert(),
        py::arg("w_vals").noconvert(),
        py::arg("w_radius").noconvert(),
        py::arg("sdfR").noconvert(),py::arg("nwalks").noconvert(),py::arg("samplevol").noconvert()
    );

    m.def(
        "compute_rot", &compute_rot,
        py::arg("A").noconvert(),
        py::arg("nx").noconvert(),
        py::arg("ny").noconvert(),
        py::arg("nz").noconvert(),
        py::arg("dx").noconvert(),
        py::arg("dy").noconvert(),
        py::arg("dz").noconvert()
    );

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
