from setuptools import setup
from setuptools import find_packages

setup(name='PGCN',
      version='0.1',
      description='Progressive Graph Convolutional Networks for Semi-Supervised Node Classification in PyTorch',
      author='Negar Heidari',
      author_email='negar.heidari@ece.au.dk',
      download_url='https://github.com/negarhdr/PGCN',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'pgcn': ['README.md']},
      packages=find_packages())
