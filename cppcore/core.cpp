#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
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
                          float sdfR, int nwalks)
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
                a_w_vals(0,i) = float(n_source_points)*fx; // Source term
                a_w_vals(1,i) = float(n_source_points)*fy; // Source term
                a_w_vals(2,i) = float(n_source_points)*fz; // Source term
                a_w_radius(i) = float(n_source_points)*ry;    // Sphere radius

                // Select random numbers for the following
                float r1 = (float)rand() / (float)RAND_MAX;
                float r2 = (float)rand() / (float)RAND_MAX;

                // Sample a new point on the sphere of center {x,y,z} and radius ry
                float theta = acos(2.0f * r1 - 1.0f);
                float phi = 2.0f * PI * r2;
                x += ry * sin(theta) * cos(phi);
                y += ry * sin(theta) * sin(phi);
                z += ry * cos(theta);
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
                          float sdfR, int nwalks)
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
                a_w_vals(i) = float(n_source_points)*fy;    // Source term
                a_w_radius(i) = float(n_source_points)*ry;    // Sphere radius

                // Select random numbers for the following
                float r1 = (float)rand() / (float)RAND_MAX;
                float r2 = (float)rand() / (float)RAND_MAX;

                // Sample a new point on the sphere of center {x,y,z} and radius ry
                float theta = acos(2.0f * r1 - 1.0f);
                float phi = 2.0f * PI * r2;
                x += ry * sin(theta) * cos(phi);
                y += ry * sin(theta) * sin(phi);
                z += ry * cos(theta);
                ry = sdfR - sqrtf(x*x + y*y + z*z);

                i++; // +1 information stored
            }
            nstarts++; // +1 walk compeleted
        }
        std::cout << "N walks started :" << nstarts << std::endl;
        return nstarts;
}

PYBIND11_MODULE(cppcore, m) {
    m.def(
        "gensourcewalks_vector", &gensourcewalks_vector,
        py::arg("source_points").noconvert(),
        py::arg("source_vectors").noconvert(),
        py::arg("w_coords").noconvert(),
        py::arg("w_vals").noconvert(),
        py::arg("w_radius").noconvert(),
        py::arg("sdfR").noconvert(),py::arg("nwalks").noconvert()
    );
    m.def(
        "gensourcewalks_scalar", &gensourcewalks_scalar,
        py::arg("source_points").noconvert(),
        py::arg("source_values").noconvert(),
        py::arg("w_coords").noconvert(),
        py::arg("w_vals").noconvert(),
        py::arg("w_radius").noconvert(),
        py::arg("sdfR").noconvert(),py::arg("nwalks").noconvert()
    );

    m.def("gensourcewalks3", [](py::array_t<float> meshpts,
                               py::array_t<float> meshtans,
                               py::array_t<float> w_coords,
                               py::array_t<float> w_vals,
                               py::array_t<float> w_radius,
                               float sdfR
    ) {
        // Array accessors (c++-python interface specificity)
        auto acc_meshpts = meshpts.unchecked<2>();
        auto acc_meshtans = meshtans.unchecked<2>();
        auto acc_w_coords = w_coords.mutable_unchecked<2>();
        auto acc_w_vals = w_vals.mutable_unchecked<2>();
        auto acc_w_radius = w_radius.mutable_unchecked<1>();

        // nwalks : we will stop until we filled nwalks datas
        py::ssize_t nwalks = acc_w_radius.shape(0);

        // Total number of meshpts
        int nmeshpts = acc_meshpts.shape(0);

        // Initialize random seed needed to sample points from the mesh's surface
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, nmeshpts);

        int i = 0;        // Nb of walk information gathered
        int nstarts = 0;  // Nb of walks started

        while (i < nwalks) {
            // Index of the surface mesh index (will define the starting pt for the walk)
            // Note that the meshpts vector is flatten : {x0,y0,z0,x1,y1,z1...}
            int start_index = dist(gen);    
            float x = acc_meshpts(start_index,0);
            float y = acc_meshpts(start_index,1);
            float z = acc_meshpts(start_index,2);

            // Source term to be propagated
            float fx = acc_meshtans(start_index,0);
            float fy = acc_meshtans(start_index,1);
            float fz = acc_meshtans(start_index,2);

            // SDF : distance to boundaries (sphere of radius sdfR)
            float ry = sdfR - sqrtf(x*x + y*y + z*z);

            // Continue the walk until we reach the boundaries or we read max info stored
            while (0.01 < ry && i < nwalks) {
                // Store current walk information
                acc_w_coords(0,i) = x;   // Position (sphere center)
                acc_w_coords(1,i) = y;   // ...
                acc_w_coords(2,i) = z;   // ...
                acc_w_vals(0,i) = float(nmeshpts)*fx; // Source term
                acc_w_vals(1,i) = float(nmeshpts)*fy; // Source term
                acc_w_vals(2,i) = float(nmeshpts)*fz; // Source term
                acc_w_radius(i) = float(nmeshpts)*ry;    // Sphere radius

                // Select random numbers for the following
                float r1 = (float)rand() / (float)RAND_MAX;
                float r2 = (float)rand() / (float)RAND_MAX;

                // Sample a new point on the sphere of center {x,y,z} and radius ry
                float theta = acos(2.0f * r1 - 1.0f);
                float phi = 2.0f * PI * r2;
                x += ry * sin(theta) * cos(phi);
                y += ry * sin(theta) * sin(phi);
                z += ry * cos(theta);
                ry = sdfR - sqrtf(x*x + y*y + z*z);

                i++; // +1 information stored
            }
            nstarts++; // +1 walk compeleted
        }
        std::cout << "N walks started :" << nstarts << std::endl;
        return nstarts;

    }, py::arg().noconvert(),   // python-c++ interface specificity
       py::arg().noconvert(),
       py::arg().noconvert(),
       py::arg().noconvert(),
       py::arg().noconvert(),
       py::arg().noconvert());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
