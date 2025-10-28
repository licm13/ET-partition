#!/usr/bin/env python

from distutils.core import setup
import TEA

setup(name='TranspirationEstimationAlgorithm',
      version=TEA.__version__,
      description='Water Flux Partitioning',
      author='Jacob A. Nelson',
      author_email='jnelson@bgc-jena.mpg.de',
      url='https://github.com/jnelson18/TranspirationEstimationAlgorithm',
      packages=['TEA']
     )
