#!/usr/bin/env python
import os
import re
# import setuptools
import setuptools.command.sdist


PYTHON_SRC = 'python'
PYTHON_FILE_RE = re.compile('.*\.(?:py[co]?)$')


setuptools.command.sdist.READMES += ('README.md',)


with open("VERSION") as f:
    version = f.read().strip()


with open('README.md') as f:
    long_description = f.read()


def find_package_datafiles(package_dir):
    """ Return a list of non-python files in package source tree

    File paths are relative to the top of the package directory provided.
    """
    # TODO: Add exclusion list/glob/regex parameter if necessary.
    non_python = set()
    for dirpath, _, fnames in os.walk(package_dir):
        non_python.update([os.path.relpath(os.path.join(dirpath, fp),
                                           package_dir)
                           for fp in fnames
                           # Things that are NOT python files
                           if PYTHON_FILE_RE.match(fp) is None])
    return list(non_python)


def list_directory_files(dirpath, exclude_dirs=(), exclude_files=()):
    """
    List files and their parent directories in the format required for the
    ``setup`` function ``data_files`` parameter:

        ...
        data_files=[
            ('dir', 'root-relative-file-path'),
            ...
        ],
        ...

    This function is intended to effectively install a directory located in the
    source root as is (e.g. the ``etc`` directory).

    :param dirpath: Base directory to start with. The directory paths returned
        start with this directory.
    :param exclude_dirs: sequence if directory paths (starting from ``dirpath``)
        that should not be included. For example, we don't want the `bin/memex'
        directory to be installed, when gathering data files for `bin`, we call
        this function like:

            list_directory_files('bin', ['bin/memex'])
    :param exclude_files: File names to ignore in directories traversed.

    """
    exclude_dirs = set(ed.strip(' /') for ed in exclude_dirs)
    exclude_files = set(ef.strip() for ef in exclude_files)
    file_paths = []
    for dirpath, dnames, fnames in os.walk(dirpath):
        # Filter out directories to be excluded
        for dn in dnames:
            if os.path.join(dirpath, dn) in exclude_dirs:
                print "skipping:", os.path.join(dirpath, dn)
                del dnames[dnames.index(dn)]
        # filter out excluded files
        fnames = set(fnames).difference(exclude_files)
        # collect directory to file paths reference
        file_paths.append(
            (dirpath, [os.path.join(dirpath, fn) for fn in fnames])
        )
    return file_paths


################################################################################


setuptools.setup(
    name='smqtk',
    version=version,  # Configure with CMake
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
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    platforms=[
        'Linux',
        'Max OS-X',
        'Unix',
        # 'Windows',  # Not tested yet
    ],

    package_dir={'': PYTHON_SRC},
    packages=setuptools.find_packages(PYTHON_SRC),
    package_data={
        'smqtk': find_package_datafiles(os.path.join(PYTHON_SRC, 'smqtk'))
    },

    # Using this for its actual purpose AS WELL AS how we're installing
    # bin-scripts, which is probably frowned on. This is a first pass until
    # scripts are restructures to use the "correct" entry_points parameter.
    data_files=(
        list_directory_files('bin',
                             exclude_dirs=['bin/memex'],
                             exclude_files=['CMakeLists.txt', 'README.txt']) +
        list_directory_files('etc')
    ),

    # use_scm_version=True,
    setup_requires=[
        'setuptools',
        # 'setuptools_scm',
    ],
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
        'scipy',
        'six',
    ],
    extras_require={
        # Various optional dependencies for plugins
        'caffe': [
            'protobuf',
            'scikit-image',
        ],
        'flann': [
            'pyflann>=1.8.4',
        ],
        'postgres': [
            'psycopg2',
        ],
        'solr': [
            'solrpy',
        ],
    },
    tests_require=[
        'coverage',
        'mock',
        'nose',
        'nose-exclude'
    ],

    # See entry_points/console_scripts as the preferred method for publishing
    #   executable scripts. Might have redesign how scripts are done if that is
    #   to be used...
    entry_points={
        'console_scripts': [
            'summarizePlugins = smqtk.bin.summarizePlugins:main'
        ]
    }
)
