from setuptools import setup, find_packages

setup(
    name="symapse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
    python_requires='>=3.8',
)
