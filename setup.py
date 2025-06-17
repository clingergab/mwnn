"""Setup configuration for Multi-Weight Neural Networks package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mwnn",
    version="0.1.0",
    author="MWNN Research Team",
    author_email="research@mwnn.org",
    description="Multi-Weight Neural Networks for Enhanced Visual Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mwnn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.931",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mwnn-train=scripts.train:main",
            "mwnn-evaluate=scripts.evaluate:main",
        ],
    },
)