from setuptools import setup, find_packages

setup(
    name='my_python_package',  # Name of the package
    version='0.1.0',
    packages=find_packages(),  # Automatically finds all sub-packages
    install_requires=[  # List any dependencies here
        'numpy',
        'pandas',
    ],
)