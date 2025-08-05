from setuptools import setup, find_packages

setup(
    name="qdotlib",
    version="0.1.0",
    author="Lucas Logue",
    author_email="llogue@nd.edu",
    description="A library for simulating and optimizing quantum gates in semiconductor quantum dots",
    
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "imageio",
        "cma",
        "scipy"
    ],
    python_requires='>=3.8',
)