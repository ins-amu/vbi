import os
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'vbi', '_version.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')


# Check for environment variable to control C++ compilation
SKIP_CPP = os.environ.get('SKIP_CPP', '').lower() in ('1', 'true', 'yes')

SRC_DIR = "vbi/models/cpp/_src"


class CustomBuildExtCommand(build_ext):
    def run(self):
        if SKIP_CPP:
            print("Skipping C++ compilation due to SKIP_CPP environment variable")
            return

        swig_files = [f for f in os.listdir(SRC_DIR) if f.endswith(".i")]
        for filename in swig_files:
            model = filename.split(".")[0]
            subprocess.check_call([
                "swig",
                "-python",
                "-c++",
                "-outdir", SRC_DIR,
                "-o", f"{SRC_DIR}/{model}_wrap.cxx",
                f"{SRC_DIR}/{model}.i",
            ])

        super().run()


def create_extension(model):
    return Extension(
        f"vbi.models.cpp._src._{model}",
        sources=[f"{SRC_DIR}/{model}_wrap.cxx"],
        include_dirs=[SRC_DIR],
        extra_compile_args=["-std=c++11", "-O2", "-fPIC", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    )


extensions = []
if not SKIP_CPP and os.path.isdir(SRC_DIR):
    for filename in os.listdir(SRC_DIR):
        if filename.endswith(".i"):
            extensions.append(create_extension(filename.split(".")[0]))


setup(
    name="vbi",
    version=get_version(),
    packages=find_packages(),
    package_data={
        "vbi": ["models/pytorch/data/*"],
        "vbi.models.cpp._src": ["*.so", "*.h", "*.i", "*.py"],
        "vbi.feature_extraction": ["*.json", "*.jar"],
    },
    ext_modules=extensions,
    cmdclass={"build_ext": CustomBuildExtCommand},
)
