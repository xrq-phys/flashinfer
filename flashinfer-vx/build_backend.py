"""
Custom build backend that compiles the attached cubins to a runner library.
"""

import glob
import os
import shutil
import sys
import tempfile
import distutils.ccompiler
import distutils.sysconfig
from pathlib import Path
from torch.utils.cpp_extension import load as load_extension
from setuptools import build_meta as _orig
import setuptools


def _cpp_compiler():
    """Get the C++ compiler to use."""
    # Initialize the C++ compiler
    compiler = distutils.ccompiler.new_compiler()
    setuptools.extension.Extension('_', sources=[]).use_2to3_fixers = None
    distutils.sysconfig.customize_compiler(compiler)

    # Get the compiler details
    return compiler.compiler_cxx[0]


def _compile_extension():
    """Compile the CUDA extension and return the path to the compiled .so file."""
    backend_dir = Path(__file__).parent

    # Get C++ compiler
    cxx = _cpp_compiler()

    # Define sources
    sources = [
        str(backend_dir / 'cpp' / 'fmhaRunner.cu'),
        str(backend_dir / 'cpp' / 'sequencePreproc.cu')
    ] + glob.glob(str(backend_dir / 'cubin' / '*_cubin.cpp'))

    # Compiler flags
    cflags = ['-std=c++20', '-fPIC', '-O3']
    cuflags = ['-std=c++20', '-ccbin', f'{cxx}', '-Xcompiler', '-fPIC', '-O3']

    # Include directories (relative to backend_dir)
    include_dirs = [
        str(backend_dir),
        str(backend_dir / 'cubin'),
        str(backend_dir.parent / 'include'),
        str(backend_dir.parent / '3rdparty' / 'cutlass' / 'include'),
        str(backend_dir.parent / '3rdparty' / 'spdlog' / 'include'),
    ]

    # Build dir
    build_dir = backend_dir / 'build' / 'cpp'
    build_dir.mkdir(exist_ok=True)

    # Build the extension
    load_extension(
        name='flashinfer_vx_cubin',
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuflags,
        extra_ldflags=['-lcuda'],
        extra_include_paths=include_dirs,
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=True,
    )

    # Find the build product
    so_files = list(build_dir.glob('flashinfer_vx_cubin*.so'))
    if not so_files:
        raise RuntimeError("Failed to find compiled .so file")

    return so_files[0]


def _install_extension():
    """Compile and install the extension to the package directory."""
    backend_dir = Path(__file__).parent
    package_dir = backend_dir / 'flashinfer_vx'

    so_file = _compile_extension()

    dest_file = package_dir / so_file.name
    shutil.copy2(so_file, dest_file)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel with CUDA extension."""
    _install_extension()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install with CUDA extension."""
    _install_extension()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


# Pass through all other hooks
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
