import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom command to compile C++ and SWIG wrapper
class CustomBuildExtCommand(build_ext):
    def run(self):
        
        swig_files = [filename for filename in os.listdir("vbi/models/cpp/_src") if filename.endswith(".i")]
        # Compile SWIG interfaces for each model
        for filename in swig_files:
            
            model = filename.split(".")[0]
            interface_file = f"vbi/models/cpp/_src/{model}.i"
            wrapper_file = f"vbi/models/cpp/_src/{model}_wrap.cxx"
            output_dir = "vbi/models/cpp/_src"
            
            subprocess.check_call(
                [
                    "swig", 
                    "-python", 
                    "-c++", 
                    "-outdir", output_dir, 
                    "-o", wrapper_file, 
                    interface_file
                ]
            )

        # Continue with the standard build_ext run
        super().run()

# Helper function to create C++ extensions for each model
def create_extension(model):
    
    return Extension(
        f"models/cpp/_src._{model}",
        sources=[f"vbi/models/cpp/_src/{model}_wrap.cxx"],
        include_dirs=[],  # Include your header directory "vbi/models/cpp/_src"
        extra_compile_args=["-O2", "-fPIC" ],  # Compilation flags
        extra_link_args=["-fopenmp"], 
    )

# Define all C++ extensions based on the models
# models = ["do", "jr_sde", "jr_sdde", "mpr_sde", "sl_sdde", "km_sde", "wc_ode", "ww_sde", "vep"]
models = []
for filename in os.listdir("vbi/models/cpp/_src"):
    if filename.endswith(".i"):
        model = filename.split(".")[0]
        models.append(model)
extensions = [create_extension(model) for model in models]

# Setup function
setup(
    name="vbi",
    version="0.1.0",
    description="A Python package with C++ integration via SWIG",
    packages=["vbi"],
    package_dir={"": "vbi"},
    ext_modules=extensions,  # Include all the C++ extensions
    cmdclass={
        "build_ext": CustomBuildExtCommand,  # Override the default build_ext
    },
)
