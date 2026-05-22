from setuptools import setup, find_packages

setup(
    name="deepal6",
    version="1.0.0",
    author="Bob Philip Aila",
    author_email="",
    description=(
        "Deep Active Learning library — 6 query strategies vs. random baseline, "
        "for tabular and image domains (AIMS Rwanda Thesis)"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bobphilip/deepal6",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "torch>=1.12",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
    ],
    extras_require={
        "image": [
            "torchvision>=0.13",
            "Pillow>=9.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
