# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("cppcore",
        ["cppcore/core.cpp"],
        include_dirs=["C:/Program Files (x86)/NanoVDB/include"],
        libraries=["C:/Program Files (x86)/NanoVDB/bin"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="walkonspheres",
    version=__version__,
    author="Vincent Sivadon",
    author_email="vincent.sivadon@laposte.net",
    # url="https://github.com/pybind/python_example",
    description="Walk On Spheres simulation tools",
    long_description="", #readme
    ext_modules=ext_modules,
    packages=find_packages(),
    install_requires = [
        "warp-lang",
        "numpy",
        "polyscope",
        "pybind11"
    ],
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
