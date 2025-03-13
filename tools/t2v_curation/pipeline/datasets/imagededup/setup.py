from setuptools import Extension, setup

MOD_NAME = "brute_force_cython_ext"
MOD_PATH = "pipeline/datasets/imagededup/handlers/search/brute_force_cython_ext"
COMPILE_LINK_ARGS = ["-O3", "-march=native", "-mtune=native"]

ext_modules = [
    Extension(
        MOD_NAME,
        [MOD_PATH + ".cpp"],
    )
]

setup(
    name="brute_force_cython_ext",
    ext_modules=ext_modules,
)
