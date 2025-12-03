"""
Custom build backend that compiles the attached cubins to a runner library.
"""

import os
import shutil
import subprocess
import distutils.ccompiler
import distutils.sysconfig
from pathlib import Path
from setuptools import build_meta as _orig
import tvm_ffi as ffi
import setuptools


def _find_cuda_home():
    """Find CUDA installation directory."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        return Path(cuda_home)

    # Try common locations
    common_paths = [
        '/usr/local/cuda',
        '/usr/lib/cuda',
        '/opt/cuda',
    ]
    for path in common_paths:
        if Path(path).exists():
            return Path(path)

    # Try to find nvcc in PATH
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc'], text=True).strip()
        return Path(nvcc_path).parent.parent
    except:
        pass

    raise RuntimeError("Could not find CUDA installation. Please set CUDA_HOME environment variable.")

def _find_python_include():
    """Find Python include directory."""
    import sysconfig
    return sysconfig.get_path('include')

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
    tvm_ffi_dir = Path(ffi.__file__).parent

    # Get C++ compiler
    cxx = _cpp_compiler()

    # Find CUDA
    cuda_home = _find_cuda_home()
    nvcc = cuda_home / 'bin' / 'nvcc'
    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found at {nvcc}")

    # Define sources
    cu_sources = [
        backend_dir / 'cpp' / 'fmhaRunner.cu',
        backend_dir / 'cpp' / 'sequencePreproc.cu'
    ]
    cpp_sources = list((backend_dir / 'cubin').glob('*_cubin.cpp'))

    # Include directories
    include_dirs = [
        str(backend_dir),
        str(backend_dir / 'cubin'),
        str(backend_dir.parent / 'include'),
        str(backend_dir.parent / '3rdparty' / 'cutlass' / 'include'),
        str(backend_dir.parent / '3rdparty' / 'spdlog' / 'include'),
        str(tvm_ffi_dir / "include"),
        str(cuda_home / 'include'),
        _find_python_include(),
    ]

    # Build dir
    build_dir = backend_dir / 'build' / 'cpp'
    build_dir.mkdir(parents=True, exist_ok=True)

    # Compiler flags
    cxx_flags = ['-std=c++20', '-fPIC', '-O3']
    nvcc_flags = [
        '-std=c++20',
        f'-ccbin={cxx}',
        '--compiler-options', '-fPIC',
        '-O3',
        '--extended-lambda',
        '--expt-relaxed-constexpr',
    ]

    # Add include directories to flags
    include_flags = [f'-I{inc}' for inc in include_dirs]

    # Object files
    object_files = []

    # Compile .cu files with nvcc
    for cu_file in cu_sources:
        obj_file = build_dir / (cu_file.stem + '.o')
        cmd = [
            str(nvcc),
            '-c',
            str(cu_file),
            '-o', str(obj_file),
        ] + nvcc_flags + include_flags

        subprocess.check_call(cmd, cwd=str(backend_dir))
        object_files.append(obj_file)

    # Compile .cpp files with g++
    for cpp_file in cpp_sources:
        obj_file = build_dir / (cpp_file.stem + '.o')
        cmd = [
            cxx,
            '-c',
            str(cpp_file),
            '-o', str(obj_file),
        ] + cxx_flags + include_flags

        subprocess.check_call(cmd, cwd=str(backend_dir))
        object_files.append(obj_file)

    # Link into shared library
    so_file = build_dir / 'flashinfer_vx_cubin.so'
    link_cmd = [cxx, '-shared', '-o', str(so_file)]
    link_cmd += [str(obj) for obj in object_files]
    link_cmd += [str(tvm_ffi_dir) / 'lib' / 'libtvm_ffi.so']
    link_cmd += [f'-L{cuda_home}/lib64', '-lcuda', '-lcudart']

    subprocess.check_call(link_cmd, cwd=str(backend_dir))

    if not so_file.exists():
        raise RuntimeError(f"Failed to create {so_file}")

    return so_file


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
