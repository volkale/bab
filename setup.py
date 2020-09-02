# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()


with open('requirements_test.txt', 'r') as f:
    tests_require = f.read().splitlines()


setup(
    name='bab',
    version='0.1.4',
    description='Bayesian estimation for A / B testing',
    author='Alexander Volkmann',
    author_email='alexv@gmx.de',
    packages=['bab'],
    package_dir={'bab': 'bab'},
    package_data={'bab': ['model.stan', 'tests/data/*.csv']},
    include_package_data=True,
    setup_requires=['pytest-runner>=4.2', 'flake8'],
    install_requires=install_requires,
    tests_require=tests_require,
    url='https://github.com/volkale/bab',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ]
)
