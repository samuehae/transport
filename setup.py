# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    # name of the package
    name='transport', 
    
    # version of package
    version='0.1.0', 
    
    # package description
    description='transport: quantum transport simulations in one dimension', 
    long_description=open('README.md').read(), 
    
    # author information
    author='Samuel Häusler', 
    url='https://github.com/samuehae/transport', 
    
    # package license
    license='MIT', 
    
    # packages to process (build, distribute, install)
    packages=['transport'], 
    
    # required packages
    install_requires=['numpy'], 
    extras_require={'examples': ['matplotlib'], }
)
