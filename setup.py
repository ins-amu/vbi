import os
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


# ----------------------------
# Policy: C++ is OPT-IN only.
# ----------------------------
def _env_true(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "on")

def _env_false(val: str) -> bool:
    return str(val).lower() in ("0", "false", "no", "off")

def cpp_opt_in_requested() -> bool:
    """
    Return True only if the user explicitly opted in to C++ builds.

    Accepted ways to opt in:
      - SKIP_CPP=0
      - VBI_ENABLE_CPP=1 (or true/yes/on)
      - VBI_CPP=1
    Anything else -> False (skip).
    """
    # Strong explicit enable
    if _env_true(os.environ.get("VBI_ENABLE_CPP", "")) or _env_true(os.environ.get("VBI_CPP", "")):
        return True

    # Backward-compatible: SKIP_CPP=0 means "do NOT skip"
    val = os.environ.get("SKIP_CPP", "")
    if val != "" and _env_false(val):
        return True

    return False


def should_build_cpp() -> bool:
    """
    - Windows: never build (always skip).
    - macOS/Linux: build only if user opted in (see cpp_opt_in_requested()).
    """
    if platform.system() == "Windows":
        return False
    return cpp_opt_in_requested()


class OptionalBuildExt(build_ext):
    """
    Build C++ extensions with graceful fallback. Since C++ is opt-in,
    this class only runs if should_build_cpp() is True.
    """

    def run(self):
        # Enforce policy early
        if not should_build_cpp():
            if platform.system() == "Windows":
                print("[VBI] Windows detected -> skipping C++ extensions (not supported).")
            else:
                print("[VBI] C++ extensions are opt-in. No opt-in detected -> skipping.")
                print("      To enable, set VBI_ENABLE_CPP=1  (or SKIP_CPP=0) before building.")
            return

        # We’re on macOS/Linux AND user opted in. Check toolchain.
        skip_reasons = []

        if not self._check_swig():
            skip_reasons.append("SWIG not found")

        if not self._check_compiler():
            skip_reasons.append("C++ compiler not found or incompatible")

        if skip_reasons:
            print(f"[VBI] Skipping C++ extensions: {', '.join(skip_reasons)}")
            print("      VBI will work with Python/NumPy/Numba models only.")
            return

        try:
            self._compile_swig_interfaces()
            super().run()
            print("[VBI] C++ extensions compiled successfully!")
        except Exception as e:
            print(f"[VBI] Failed to compile C++ extensions: {e}")
            print("      Falling back to Python/NumPy/Numba implementations.")

    def _check_swig(self):
        try:
            subprocess.run(['swig', '-version'], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _check_compiler(self):
        try:
            subprocess.run(['g++', '--version'], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _compile_swig_interfaces(self):
        src_dir = "vbi/models/cpp/_src"
        if not os.path.isdir(src_dir):
            return
        swig_files = [f for f in os.listdir(src_dir) if f.endswith(".i")]
        for swig_file in swig_files:
            model = swig_file.split(".")[0]
            interface_file = os.path.join(src_dir, f"{model}.i")
            wrapper_file = os.path.join(src_dir, f"{model}_wrap.cxx")
            cmd = [
                "swig", "-c++", "-python", "-shadow",
                "-outdir", src_dir,
                "-o", wrapper_file,
                interface_file
            ]
            subprocess.run(cmd, check=True)


def get_compile_args():
    if platform.system() == "Windows":
        return ["/O2", "/openmp", "/std:c++11", "/EHsc"]
    else:
        return [
            "-std=c++11",
            "-O2",
            "-fPIC",
            "-fopenmp",
            "-march=native",
            "-fno-strict-aliasing",
            "-Wno-sign-compare",
            "-Wno-unused-variable",
            "-Wno-reorder",
        ]


def get_link_args():
    if platform.system() == "Windows":
        return []
    else:
        return ["-fopenmp"]


def create_extension(model):
    src_dir = "vbi/models/cpp/_src"
    return Extension(
        f"vbi.models.cpp._src._{model}",
        sources=[os.path.join(src_dir, f"{model}_wrap.cxx")],
        include_dirs=[src_dir],
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
        language="c++",
    )


def get_extensions():
    # Only create extensions when actually building C++ (opt-in + non-Windows)
    if not should_build_cpp():
        msg = "[VBI] Skipping C++ extensions"
        if platform.system() == "Windows":
            msg += " on Windows."
        else:
            msg += " (no opt-in). Set VBI_ENABLE_CPP=1 or SKIP_CPP=0 to enable."
        print(msg)
        return []

    src_dir = "vbi/models/cpp/_src"
    if not os.path.exists(src_dir):
        return []

    exts = []
    for filename in os.listdir(src_dir):
        if filename.endswith(".i"):
            model = filename.split(".")[0]
            exts.append(create_extension(model))
    return exts


def get_package_data():
    base_data = {"vbi": ["models/pytorch/data/*"]}
    if should_build_cpp():
        base_data["vbi.models.cpp._src"] = ["*.so", "*.dll", "*.pyd", "*.h", "*.i", "*.py"]
    else:
        base_data["vbi.models.cpp._src"] = ["*.h", "*.i", "*.py"]
    return base_data


setup(
    name="vbi",
    version=get_version(),
    description="Virtual brain inference with optional C++ acceleration (opt-in)",
    packages=find_packages(),
    package_data=get_package_data(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
    zip_safe=False,
)
