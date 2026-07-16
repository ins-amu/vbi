import os
import sys
import shutil
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
    Build SWIG C++ extensions with graceful fallback on any failure.

    When this class is invoked, should_skip_cpp() has already returned False
    (tools are available). But compilation can still fail (e.g. compiler flags,
    missing headers); we catch that and install without C++ rather than erroring.
    """

    def run(self):
        # Double-check: should_skip_cpp() is already called in get_extensions(),
        # so this is just a safety net for edge cases.
        skip, reason = should_skip_cpp()
        if skip:
            print(f"[vbi] Skipping SWIG C++ models: {reason}")
            return

        try:
            self._compile_swig_interfaces()
            super().run()
            print("[vbi] SWIG C++ extensions compiled successfully.")
        except Exception as e:
            print(f"[vbi] SWIG C++ compilation failed: {e}")
            print("[vbi] Installing without C++ models. "
                  "Set FORCE_CPP=1 to treat this as a fatal error.")

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

SRC_DIR = "vbi/models/cpp/_src"


def create_extension(model):
    return Extension(
        f"vbi.models.cpp._src._{model}",
        sources=[f"{SRC_DIR}/{model}_wrap.cxx"],
        include_dirs=[SRC_DIR],
        extra_compile_args=["-std=c++11", "-O2", "-fPIC", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    )


def should_skip_cpp():
    """
    Return (skip: bool, reason: str).

    Decision order:
    1. Windows → always skip (no SWIG toolchain).
    2. Explicit opt-out: env var  SKIP_CPP / VBI_NO_CPP / ...  or  .skip_cpp file.
    3. Explicit opt-in:  env var  FORCE_CPP=1  → compile even if tools look absent.
    4. Auto-detect:      SWIG not found  OR  no C++ compiler → skip with a notice.
    """
    if platform.system() == "Windows":
        return True, "Windows (SWIG toolchain not supported)"

    # Explicit opt-out
    for var in ('SKIP_CPP', 'VBI_NO_CPP', 'VBI_SKIP_CPP', 'NO_CPP',
                'DISABLE_CPP', 'CPP_DISABLE'):
        if os.environ.get(var, '').lower() in ('1', 'true', 'yes', 'on'):
            return True, f"{var} environment variable"

    skip_file = os.path.join(os.path.dirname(__file__), '.skip_cpp')
    if os.path.exists(skip_file):
        return True, ".skip_cpp file present"

    # Explicit opt-in overrides auto-detection
    if os.environ.get('FORCE_CPP', '').lower() in ('1', 'true', 'yes', 'on'):
        return False, ""

    # Auto-detect: require both SWIG and a C++ compiler
    if not shutil.which("swig"):
        return True, "swig not found (install: apt install swig  or  conda install swig)"
    if not any(shutil.which(c) for c in ("g++", "c++", "clang++")):
        return True, "no C++ compiler found (install: apt install build-essential)"

    return False, ""


def get_extensions():
    """Return C++ extension list, or [] if tools are absent / skipped."""
    skip, reason = should_skip_cpp()
    if skip:
        if reason:
            print(f"[vbi] Skipping SWIG C++ models: {reason}")
            print("[vbi]   NumPy / Numba backends are fully functional without C++.")
        return []
    
    src_dir = "vbi/models/cpp/_src"
    if not os.path.exists(src_dir):
        return []
    
    extensions = []
    for filename in os.listdir(src_dir):
        if filename.endswith(".i"):
            extensions.append(create_extension(filename.split(".")[0]))


def get_package_data():
    """Get package data, excluding .so files when C++ compilation is skipped."""
    skip, _ = should_skip_cpp()
    base_data = {
        "vbi": ["models/pytorch/data/*"],
        "vbi.feature_extraction": ["*.json", "*.jar"],
    }
    if skip:
        base_data["vbi.models.cpp._src"] = ["*.h", "*.i", "*.py"]
    else:
        base_data["vbi.models.cpp._src"] = ["*.so", "*.dll", "*.pyd", "*.h", "*.i", "*.py"]
    return base_data


# Main setup
if __name__ == "__main__":
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
