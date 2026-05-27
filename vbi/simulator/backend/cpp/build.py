"""
Build-cache and CMake/compiler invocation for the VBI C++ backend.

Cache key = SHA-256 of SimulationSpec.cache_key() — same model+integrator+n_nodes
→ same .so.  Parameter sweeps that change only G/eta reuse the compiled binary.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

from vbi.simulator.spec.simulation import SimulationSpec

from .codegen import render_sim_module, render_bindings, render_cmake

_TEMPLATES_DIR = Path(__file__).resolve().parent / "_src"
_CACHE_DIR = Path(os.environ.get("VBI_CPP_CACHE",
                                  Path.home() / ".cache" / "vbi" / "cpp"))

# In-process loaded-module cache  {cache_key: module}
_MODULE_CACHE: dict[str, types.ModuleType] = {}


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

class CppBackendUnavailable(RuntimeError):
    """Raised when the C++ backend cannot be used due to missing dependencies."""


def check_prerequisites() -> None:
    """
    Verify all build-time dependencies are present before touching the filesystem.

    Raises CppBackendUnavailable with a consolidated list of what is missing
    and how to fix it.
    """
    missing: list[str] = []
    hints:   list[str] = []

    # 1. mako (template engine)
    try:
        import mako  # noqa: F401
    except ImportError:
        missing.append("mako")
        hints.append("  pip install mako")

    # 2. pybind11 (Python/C++ bindings)
    try:
        import pybind11  # noqa: F401
    except ImportError:
        missing.append("pybind11")
        hints.append("  pip install pybind11")

    # 3. A C++ compiler — accept cmake, c++, g++, or clang++ in PATH
    compiler_found = any(shutil.which(cmd) for cmd in ("cmake", "c++", "g++", "clang++"))
    if not compiler_found:
        missing.append("C++ compiler (cmake / g++ / clang++)")
        hints.append(
            "  Ubuntu/Debian : sudo apt install build-essential cmake\n"
            "  macOS         : xcode-select --install  (provides clang++)\n"
            "  Conda         : conda install -c conda-forge cxx-compiler cmake"
        )

    if missing:
        raise CppBackendUnavailable(
            "VBI C++ backend is missing required dependencies:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nTo fix:\n"
            + "\n".join(hints)
            + "\n\nAlternatively, use backend='numpy' or backend='numba'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_or_load(spec: SimulationSpec, verbose: bool = False) -> types.ModuleType:
    """
    Return the compiled pybind11 extension module for `spec`.

    - On first call: generate C++ sources, compile, cache the .so.
    - On subsequent calls (same spec): load from in-process cache or disk.

    Raises CppBackendUnavailable if mako, pybind11, or a C++ compiler is absent.
    """
    check_prerequisites()

    # Hash all build-affecting files so any change invalidates cached binaries.
    import hashlib as _hl
    _tmpl_hash = _hl.sha256(
        (_TEMPLATES_DIR / "sim_module.cpp.mako").read_bytes() +
        (_TEMPLATES_DIR / "bindings.cpp.mako").read_bytes() +
        (_TEMPLATES_DIR / "cmake_template.mako").read_bytes() +
        (_TEMPLATES_DIR / "runtime.hpp").read_bytes()
    ).hexdigest()[:12]
    key = spec.cache_key() + "_" + _tmpl_hash

    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    build_dir = _CACHE_DIR / key
    so_path   = _discover_so(build_dir, key)

    if so_path is None:
        _write_sources(spec, key, build_dir, verbose=verbose)
        so_path = _compile(build_dir, key, verbose=verbose)

    mod = _load_so(so_path, key)
    _MODULE_CACHE[key] = mod
    return mod


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------

def _write_sources(spec: SimulationSpec, key: str,
                   build_dir: Path, verbose: bool = False) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)

    module_name      = f"vbi_cpp_{key[:16]}"
    cpp_filename     = f"{module_name}.cpp"
    bindings_filename= f"{module_name}_bindings.cpp"
    runtime_dst      = build_dir / "runtime.hpp"

    # Copy shared runtime header
    shutil.copy2(_TEMPLATES_DIR / "runtime.hpp", runtime_dst)

    # Render and write generated sources
    (build_dir / cpp_filename).write_text(
        render_sim_module(spec, key), encoding="utf-8")
    (build_dir / bindings_filename).write_text(
        render_bindings(spec, module_name, cpp_filename), encoding="utf-8")
    (build_dir / "CMakeLists.txt").write_text(
        render_cmake(module_name, bindings_filename), encoding="utf-8")

    if verbose:
        print(f"[vbi-cpp] Sources written → {build_dir}")


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _compile(build_dir: Path, key: str, verbose: bool = False) -> Path:
    cmake_err: str = ""
    try:
        return _cmake_build(build_dir, verbose=verbose)
    except FileNotFoundError:
        cmake_err = "cmake not found in PATH"
    except subprocess.CalledProcessError as exc:
        cmake_err = (exc.stderr or exc.stdout or "").strip()

    if verbose:
        print(f"[vbi-cpp] CMake failed ({cmake_err}) — trying direct compiler")

    try:
        return _direct_build(build_dir, key, verbose=verbose)
    except FileNotFoundError:
        raise CppBackendUnavailable(
            "No C++ compiler found in PATH (tried cmake, c++, g++, clang++).\n"
            "Install one and retry:\n"
            "  Ubuntu/Debian : sudo apt install build-essential cmake\n"
            "  macOS         : xcode-select --install\n"
            "  Conda         : conda install -c conda-forge cxx-compiler cmake"
        ) from None
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise CppBackendUnavailable(
            f"C++ compilation failed.\n"
            f"cmake error : {cmake_err or '(not tried)'}\n"
            f"direct error: {detail}\n\n"
            f"Sources are in: {build_dir}\n"
            f"Re-run with verbose=True for full output."
        ) from exc


def _cmake_build(build_dir: Path, verbose: bool = False) -> Path:
    cmake_dir = build_dir / "cmake-build"
    cmake_dir.mkdir(exist_ok=True)

    kw: dict = {"check": True, "cwd": str(build_dir), "text": True}
    if not verbose:
        kw["capture_output"] = True

    subprocess.run(
        ["cmake", "-S", str(build_dir), "-B", str(cmake_dir),
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DPython3_EXECUTABLE={sys.executable}",
         f"-DPYTHON_EXECUTABLE={sys.executable}"],
        **kw)
    subprocess.run(
        ["cmake", "--build", str(cmake_dir), "--config", "Release"],
        **kw)

    so = _discover_so(build_dir, _module_name(build_dir))
    if so is None:
        # cmake may have placed it inside cmake-build
        so = _discover_so(cmake_dir, _module_name(build_dir))
    if so is None:
        raise FileNotFoundError(f"Built .so not found in {build_dir}")
    # Move to build_dir top level for consistent discovery
    if so.parent != build_dir:
        dst = build_dir / so.name
        shutil.copy2(so, dst)
        so = dst
    if verbose:
        print(f"[vbi-cpp] CMake built → {so}")
    return so


def _direct_build(build_dir: Path, key: str, verbose: bool = False) -> Path:
    module_name      = f"vbi_cpp_{key[:16]}"
    bindings_filename= f"{module_name}_bindings.cpp"

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path    = build_dir / f"{module_name}{ext_suffix}"

    include_flags = _include_flags()
    ld_flags      = _ld_flags()

    _env_flags = os.environ.get("VBI_CPP_CXXFLAGS", "")
    opt_flags = _env_flags.split() if _env_flags else ["-O3", "-march=native", "-ffast-math"]
    cmd = [
        "c++", *opt_flags,
        "-shared", "-fPIC", "-std=c++17",
        *include_flags,
        str(build_dir / bindings_filename),
        "-o", str(so_path),
        *ld_flags,
    ]
    if verbose:
        print(f"[vbi-cpp] Direct build: {' '.join(cmd)}")

    kw: dict = {"check": True, "cwd": str(build_dir), "text": True}
    if not verbose:
        kw["capture_output"] = True
    subprocess.run(cmd, **kw)

    if verbose:
        print(f"[vbi-cpp] Direct built → {so_path}")
    return so_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_so(so_path: Path, key: str) -> types.ModuleType:
    # Module name must match the PYBIND11_MODULE(...) name compiled into the .so
    mod_name = f"vbi_cpp_{key[:16]}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, so_path)
    mod      = importlib.util.module_from_spec(spec_obj)   # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec_obj.loader.exec_module(mod)                        # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_so(directory: Path, module_name: str) -> Path | None:
    if not directory.exists():
        return None
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        p = directory / f"{module_name}{suffix}"
        if p.exists():
            return p
    for p in sorted(directory.glob(f"{module_name}*.so")):
        return p
    return None


def _module_name(build_dir: Path) -> str:
    # Recover from CMakeLists.txt project() line
    cmake = build_dir / "CMakeLists.txt"
    if cmake.exists():
        for line in cmake.read_text().splitlines():
            line = line.strip()
            if line.startswith("project("):
                return line.split("(")[1].split(")")[0].split()[0]
    return build_dir.name


def _include_flags() -> list[str]:
    flags: list[str] = []
    for key in ("include", "platinclude"):
        d = sysconfig.get_paths().get(key)
        if d and Path(d).exists():
            flags.append(f"-I{d}")
    flags.append(f"-I{_pybind11_include()}")
    return flags


def _pybind11_include() -> str:
    env = os.environ.get("PYBIND11_INCLUDE_DIR")
    if env and Path(env).exists():
        return env
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        pass
    raise RuntimeError(
        "pybind11 not found. Install it: pip install pybind11  "
        "or set PYBIND11_INCLUDE_DIR.")


def _ld_flags() -> list[str]:
    flags: list[str] = []
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        flags.append(f"-L{libdir}")
    ver      = sysconfig.get_config_var("VERSION") or ""
    abiflag  = sysconfig.get_config_var("ABIFLAGS") or ""
    if ver:
        flags.append(f"-lpython{ver}{abiflag}")
    for var in ("LIBS", "SYSLIBS", "LINKFORSHARED"):
        val = sysconfig.get_config_var(var)
        if val:
            flags.extend(shlex.split(val))
    return flags
