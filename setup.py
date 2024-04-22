# from sbi_nmms.__init__ import __version__ as v
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="vbi",
    version="0.1",
    author="Abolfazl ziaeemehr, Meysam Hashemi",
    author_email="a.ziaeemehr@gmail.com, meysam.hashemi@gmail.com",
    description="Virtual brain inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ziaeemehr/vbi_paper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.9',
    # packages=['vbi'],
    # package_dir={'vbi': 'vbi'},
    # package_data={'vbi': ['CPPModels/*.so']},
    # install_requires=requirements,
    extra_require={
        "dev": [
            "pre-commit",
            "nbformat",
            "nbconvert",
            "ruff",
        ],
        "all": [
            'jax',
            'pycatch22',
            "pre-commit",
            "nbformat",
            "nbconvert",
            "ruff",
            ],
    },
    # include_package_data=True,
)
