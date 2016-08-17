#!/usr/bin/env python
from distutils.core import setup
import os.path as osp
import setuptools


# Replace with CMake configuration
PYTHON_SRC = osp.abspath(osp.join(osp.dirname(__file__), 'python'))


# TODO: Get long description from the README.md file in project source dir
long_description = "WIP"


setup(
    name='smqtk',
    version='0.6.2',  # Configure with CMake
    description='Python toolkit for pluggable algorithms and data structures '
                'for multimedia-based machine learning',
    long_description=long_description,
    author='Kitware, Inc.',
    author_email='smqtk-developers@kitware.com',
    url='https://github.com/Kitware/SMQTK',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],

    # '' refers to base working directory?
    package_dir={'': PYTHON_SRC},
    packages=setuptools.find_packages(PYTHON_SRC),
    # package_data={
    #     'smqtk': [
    #         # list of non-python files relative to package root
    #     ]
    # },

    install_requires=[
        'flask',
        'flask-basicauth',
        'flask-login',
        'imageio',
        'jinja2',
        'matplotlib',
        'numpy',
        'pillow',
        'pymongo',
        'requests',
        'scikit-learn',
        'scipy'
    ],
    extra_require={
        'test': [
            'coverage',
            'mock',
            'nose',
            'nose-exclude'
        ],
        # Various optional dependencies for plugins
        'caffe': [
            'protobuf',
            'scikit-image',
        ],
        'flann': [
            'pyflann',
        ],
        'postgres': [
            'psycopg2',
        ],
        'solr': [
            'solrpy',
        ],
    },

    # ``scripts=[...]``: list of filepaths to install as scripts. This ends up
    #   not working because what is installed is a thin shell that contains the
    #   absolute path provided here, which will not work on other people's
    #   systems.
    # See entry_points/console_scripts as the preferred method for publishing
    #   executable scripts. Might have redesign how scripts are done if that is
    #   to be used...
)
