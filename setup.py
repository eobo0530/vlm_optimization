from setuptools import setup, find_packages

setup(
    name="vlm_hybrid",
    version="0.1.0",
    description="Integrated Hybrid VLM (DyMU + FastV) for LLaVA",
    author="VLM Optimization Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Dependencies are managed via setup_hybrid.sh and requirements_dist.txt
        # for complex local path resolution.
    ],
)
