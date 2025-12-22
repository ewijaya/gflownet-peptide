"""Setup script for gflownet-peptide package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gflownet-peptide",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="Diverse therapeutic peptide generation with GFlowNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/gflownet-peptide",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fair-esm>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "umap-learn>=0.5.0",
            "hdbscan>=0.8.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gflownet-train=scripts.train_gflownet:main",
            "gflownet-sample=scripts.sample:main",
            "gflownet-eval=scripts.evaluate:main",
        ],
    },
)
