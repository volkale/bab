# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='bab',
    version='0.1.0',
    description='Bayesian estimation for A / B testing',
    author='Alexander Volkmann',
    author_email='alexv@gmx.de',
    packages=['bab'],
    package_dir={'bab': 'bab'},
    package_data={'bab': ['tests/data/*.csv']},
    url='https://github.com/volkale/bab',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ]

)
