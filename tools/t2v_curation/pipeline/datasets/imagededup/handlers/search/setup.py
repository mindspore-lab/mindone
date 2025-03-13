from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "brute_force_cython_ext",  # This should be the module name without the directory structure
        ["imagededup/handlers/search/brute_force_cython_ext.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],  # or -std=c++14, -std=c++17 as required
    )
]

setup(
    name="brute_force_cython_ext",
    ext_modules=cythonize(extensions),
)
