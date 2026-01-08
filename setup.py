"""
Setup script for Torch ROSA C++ Extension.
"""

import os
from pathlib import Path
from typing import List, Union

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# Constants
LIBRARY_NAME = "rosa"
PACKAGE_VERSION = "0.0.1"
PACKAGE_DESCRIPTION = "Torch ROSA C++ Extension"


def get_extensions() -> List[Union[CUDAExtension, CppExtension]]:
    """
    Retrieve the C++/CUDA extensions to be built.

    Returns:
        List[Union[CUDAExtension, CppExtension]]: A list of extension modules.
    """
    use_cuda = (
        os.getenv("USE_CUDA", "1") == "1"
        and torch.cuda.is_available()
        and CUDA_HOME is not None
    )

    extension_class = CUDAExtension if use_cuda else CppExtension

    extra_compile_args = {
        "cxx": ["-O3", "-fopenmp", "-fdiagnostics-color=always"],
        "nvcc": ["-O3"],
    }
    extra_link_args = []

    root_dir = Path(__file__).parent
    extensions_dir = root_dir / LIBRARY_NAME / "csrc"
    extensions_cuda_dir = extensions_dir / "cuda"

    sources = [str(p) for p in extensions_dir.glob("*.cpp")]

    if use_cuda:
        cuda_sources = [str(p) for p in extensions_cuda_dir.glob("*.cu")]
        sources.extend(cuda_sources)

    ext_modules = [
        extension_class(
            f"{LIBRARY_NAME}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    setup(
        name=LIBRARY_NAME,
        version=PACKAGE_VERSION,
        packages=find_packages(),
        ext_modules=get_extensions(),
        install_requires=["torch"],
        description=PACKAGE_DESCRIPTION,
        cmdclass={"build_ext": BuildExtension},
    )
