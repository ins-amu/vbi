import os
import sys
import platform
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'vbi', '_version.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')


class OptionalBuildExt(build_ext):
    """
    Build C++ extensions with graceful fallback on failure.
    Supports environment variables for controlling compilation.
    """
    
    def run(self):
        # Automatically skip C++ compilation on Windows
        if platform.system() == "Windows":
            print("Detected Windows system - skipping C++ extensions automatically")
            print("VBI will work with Python/NumPy/Numba models only.")
            return
        
        # Check for skip flags on non-Windows systems
        skip_reasons = []
        
        # Check multiple environment variables for flexibility
        skip_env_vars = [
            'SKIP_CPP', 'VBI_NO_CPP', 'VBI_SKIP_CPP', 
            'NO_CPP', 'DISABLE_CPP', 'CPP_DISABLE'
        ]
        
        for env_var in skip_env_vars:
            if os.environ.get(env_var, '').lower() in ('1', 'true', 'yes', 'on'):
                skip_reasons.append(f"{env_var} environment variable set")
                break  # Only report the first one found
        
        # Check for special marker file (alternative to env vars)
        skip_file = os.path.join(os.path.dirname(__file__), '.skip_cpp')
        if os.path.exists(skip_file):
            skip_reasons.append(".skip_cpp file found")
        
        # Check for required tools
        if not self._check_swig():
            skip_reasons.append("SWIG not found")
        
        if not self._check_compiler():
            skip_reasons.append("C++ compiler not found or incompatible")
        
        if skip_reasons:
            print(f"Skipping C++ extensions: {', '.join(skip_reasons)}")
            print("VBI will work with Python/NumPy/Numba models only.")
            print("To force skipping C++ compilation, set SKIP_CPP=1 or VBI_NO_CPP=1")
            return
        
        try:
            self._compile_swig_interfaces()
            super().run()
            print("C++ extensions compiled successfully!")
        except Exception as e:
            print(f"Failed to compile C++ extensions: {e}")
            print("VBI will work with Python/NumPy/Numba models only.")
            print("To force skipping C++ compilation, set SKIP_CPP=1 or VBI_NO_CPP=1")
    
    def _check_swig(self):
        try:
            subprocess.run(['swig', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_compiler(self):
        try:
            # Check for gcc/g++ (non-Windows systems only)
            subprocess.run(['g++', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _compile_swig_interfaces(self):
        src_dir = "vbi/models/cpp/_src"
        swig_files = [f for f in os.listdir(src_dir) if f.endswith(".i")]
        
        for swig_file in swig_files:
            model = swig_file.split(".")[0]
            interface_file = os.path.join(src_dir, f"{model}.i")
            
            # Use .cxx extension to match what SWIG generates and what the error shows
            wrapper_file = os.path.join(src_dir, f"{model}_wrap.cxx")
            
            cmd = [
                "swig", 
                "-c++", 
                "-python", 
                "-shadow",  # Add shadow flag like in makefile
                "-outdir", src_dir, 
                "-o", wrapper_file, 
                interface_file
            ]
            
            subprocess.run(cmd, check=True)


def get_compile_args():
    """Get platform-specific compile arguments."""
    if platform.system() == "Windows":
        # MSVC flags - conservative optimization
        return ["/O2", "/openmp", "/std:c++11", "/EHsc"]
    else:
        # GCC flags - safe optimizations for numerical code
        return [
            "-std=c++11",              # C++11 standard
            "-O2",                     # Safe optimization level
            "-fPIC",                   # Position independent code
            "-fopenmp",                # OpenMP support
            "-march=native",           # Optimize for current CPU architecture
            "-fno-strict-aliasing",    # Avoid pointer aliasing issues (safer for numerical code)
            "-Wno-sign-compare",       # Suppress signed/unsigned comparison warnings (we'll fix critical ones manually)
            "-Wno-unused-variable",    # Suppress unused variable warnings (safe to ignore)
            "-Wno-reorder"             # Suppress member initialization order warnings (safe to ignore)
        ]


def get_link_args():
    """Get platform-specific link arguments."""
    if platform.system() == "Windows":
        return []  # OpenMP linking is handled automatically on Windows with /openmp
    else:
        return ["-fopenmp"]


def create_extension(model):
    """Create a C++ extension for a model with platform-specific settings."""
    src_dir = "vbi/models/cpp/_src"
    
    return Extension(
        f"vbi.models.cpp._src._{model}",
        sources=[os.path.join(src_dir, f"{model}_wrap.cxx")],  # Use .cxx extension
        include_dirs=[src_dir],
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
        language="c++"
    )


def should_skip_cpp():
    """Check if C++ compilation should be skipped."""
    # Automatically skip C++ compilation on Windows
    if platform.system() == "Windows":
        return True
    
    skip_env_vars = [
        'SKIP_CPP', 'VBI_NO_CPP', 'VBI_SKIP_CPP', 
        'NO_CPP', 'DISABLE_CPP', 'CPP_DISABLE'
    ]
    
    # Check environment variables
    for env_var in skip_env_vars:
        if os.environ.get(env_var, '').lower() in ('1', 'true', 'yes', 'on'):
            return True
    
    # Check for skip file
    skip_file = os.path.join(os.path.dirname(__file__), '.skip_cpp')
    if os.path.exists(skip_file):
        return True
    
    return False


def get_extensions():
    """Get list of C++ extensions if compilation is not skipped."""
    if should_skip_cpp():
        if platform.system() == "Windows":
            print("Skipping C++ extensions on Windows - using Python/NumPy/Numba models only")
        else:
            print("Skipping C++ extensions due to environment variable")
        return []
    
    src_dir = "vbi/models/cpp/_src"
    if not os.path.exists(src_dir):
        return []
    
    extensions = []
    for filename in os.listdir(src_dir):
        if filename.endswith(".i"):
            model = filename.split(".")[0]
            extensions.append(create_extension(model))
    
    return extensions


def get_package_data():
    """Get package data, excluding .so files if C++ is skipped."""
    base_data = {
        "vbi": ["models/pytorch/data/*"],
    }
    
    if should_skip_cpp():
        # Exclude compiled extensions when skipping C++
        base_data["vbi.models.cpp._src"] = ["*.h", "*.i", "*.py"]
        print("Excluding C++ compiled files from package data")
    else:
        # Include compiled extensions when C++ is enabled
        base_data["vbi.models.cpp._src"] = ["*.so", "*.dll", "*.pyd", "*.h", "*.i", "*.py"]
    
    return base_data


# Main setup
setup(
    name="vbi",
    version=get_version(),
    description="Virtual brain inference with optional C++ acceleration",
    packages=find_packages(),
    package_data=get_package_data(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
    zip_safe=False,  # Important for C extensions
)
