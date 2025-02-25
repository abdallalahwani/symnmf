# File: setup.py

from setuptools import setup, Extension

# Define the C extension module
symnmf_module = Extension(
    'symnmfmodule',  # Module name
    sources=['symnmfmodule.c', 'symnmf.c'],  # Source files
    include_dirs=[],  # Include directories if needed
    extra_compile_args=['-ansi', '-Wall', '-Wextra', '-Werror', '-pedantic-errors'],  # Compilation flags
)

# Setup function to build the module
setup(
    name='symnmf',
    version='1.0',
    description='Symmetric Non-negative Matrix Factorization (SymNMF) module',
    ext_modules=[symnmf_module],
)
