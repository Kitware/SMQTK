#!/usr/bin/env python
import ast
import os
from pathlib import Path
import pkg_resources
# noinspection PyUnresolvedReferences
from pkg_resources.extern import packaging
import re
import setuptools
from typing import cast, Generator, Iterable, List, Optional, Tuple, Union
import urllib.parse


###############################################################################
# Some helper functions

def parse_version(fpath: Union[str, Path]) -> str:
    """
    Statically parse the "__version__" number string from a python file.

    TODO: Auto-append dev version based on how forward from latest release
          Basically a simpler version of what setuptools_scm does but without
          the added cruft and bringing the ENTIRE git repo in with the dist
          See: https://github.com/pypa/setuptools_scm/blob/master/setuptools_scm/version.py
          Would need to know number of commits ahead from last version tag.
    """
    with open(fpath, 'r') as file_:
        pt = ast.parse(file_.read())

    class VersionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.version: Optional[str] = None

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    self.version = cast(ast.Str, node.value).s

    visitor = VersionVisitor()
    visitor.visit(pt)
    if visitor.version is None:
        raise RuntimeError("Failed to find __version__!")
    return visitor.version


def parse_req_strip_version(filepath: Union[str, Path]) -> List[str]:
    """
    Read requirements file and return the list of requirements specified
    therein but with their version aspects striped.

    See pkg_resources.Requirement docs here:
        https://setuptools.readthedocs.io/en/latest/pkg_resources.html#requirement-objects
    """
    filepath = Path(filepath)
    # Known prefixes of lines that are definitely not requirements
    # specifications.
    skip_prefix_tuple = (
        "#", "--index-url"
    )

    def filter_req_lines(_filepath: Path) -> Generator[str, None, None]:
        """ Filter lines from file that are requirements. """
        with open(_filepath, 'r') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith(skip_prefix_tuple):
                    # Empty or has a skippable prefix.
                    continue
                elif _line.startswith('-r '):
                    # sub-requirements file specification, yield that file's
                    # req lines.
                    target = _filepath.parent / _line.split(" ")[1]
                    for _r_line in filter_req_lines(target):
                        yield _r_line
                elif _line.startswith('-e '):
                    # Indicator for URL-based requirement. Look to the egg
                    # fragment.
                    frag = urllib.parse.urlparse(_line.split(' ')[1]).fragment
                    try:
                        egg = dict(
                            cast(Tuple[str, str], part.split('=', 1))
                            for part in frag.split('&')
                            if part  # handle no fragments
                        )['egg']
                    except KeyError:
                        raise packaging.requirements.InvalidRequirement(
                            f"Failed to parse egg name from the requirements "
                            f"line: '{_line}'"
                        )
                    yield egg
                else:
                    yield _line

    def strip_req_specifier(
        req_iter: Iterable[pkg_resources.Requirement]
    ) -> Generator[pkg_resources.Requirement, None, None]:
        """
        Modify requirements objects to null out the specifier component.
        """
        for r in req_iter:
            r.specs = []
            # `specifier` property is defined in extern base-class of the
            # `pkg_resources.Requirement` type.
            # noinspection PyTypeHints
            r.specifier = packaging.specifiers.SpecifierSet()  # type: ignore
            yield r

    return [
        str(req)
        for req in strip_req_specifier(
            pkg_resources.parse_requirements(filter_req_lines(filepath))
        )
    ]


PYTHON_FILE_RE = re.compile(r'.*\.(?:py[co]?)$')


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
                print("skipping:", os.path.join(dirpath, dn))
                del dnames[dnames.index(dn)]
        # filter out excluded files
        fnames = set(fnames).difference(exclude_files)
        # collect directory to file paths reference
        file_paths.append(
            (dirpath, [os.path.join(dirpath, fn) for fn in fnames])
        )
    return file_paths


################################################################################

PYTHON_SRC = 'python'
PACKAGE_NAME = "smqtk"
SETUP_DIR = Path(__file__).parent

with open(SETUP_DIR / "README.md") as f:
    LONG_DESCRIPTION = f.read()

VERSION = parse_version(SETUP_DIR / PYTHON_SRC / PACKAGE_NAME / "__init__.py")


if __name__ == "__main__":
    setuptools.setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description='Python toolkit for pluggable algorithms and data structures '
                    'for multimedia-based machine learning',
        long_description=LONG_DESCRIPTION,
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
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
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
        data_files=list_directory_files('etc'),
        # Required for mypy to be able to find the installed package.
        # https://mypy.readthedocs.io/en/latest/installed_packages.html#installed-packages
        zip_safe=False,

        install_requires=parse_req_strip_version(SETUP_DIR / "requirements" / "runtime.txt"),
        extras_require={
            # Various optional dependencies for plugins
            'docs': parse_req_strip_version(SETUP_DIR / "requirements" / "docs.txt"),
            'caffe': [
                'protobuf',
                'scikit-image',
            ],
            'magic': [
                'file-magic',
            ],
            'flann': [
                'pyflann>=1.8.4',
            ],
            'libmagic': [
                'file-magic',
            ],
            'postgres': [
                'psycopg2-binary',
            ],
            'solr': [
                'solrpy',
            ],
            'girder': [
                'girder-client',
            ],
        },

        # See entry_points/console_scripts as the preferred method for publishing
        #   executable scripts. Might have redesign how scripts are done if that is
        #   to be used...
        # TODO: Rename camel-case scripts to ``smqtk-...`` format without camel-case
        entry_points={
            'smqtk_plugins': [
                # Included batteries!
                # Algorithms
                "smqtk.algorithms.classifier._plugins = smqtk.algorithms.classifier._plugins",
                "smqtk.algorithms.descriptor_generator._plugins = smqtk.algorithms.descriptor_generator._plugins",
                "smqtk.algorithms.image_io._plugins = smqtk.algorithms.image_io._plugins",
                "smqtk.algorithms.nn_index._plugins = smqtk.algorithms.nn_index._plugins",
                "smqtk.algorithms.nn_index.hash_index._plugins = smqtk.algorithms.nn_index.hash_index._plugins",
                "smqtk.algorithms.nn_index.lsh.functors._plugins = smqtk.algorithms.nn_index.lsh.functors._plugins",
                "smqtk.algorithms.rank_relevancy._plugins = smqtk.algorithms.rank_relevancy._plugins",
                "smqtk.algorithms.relevancy_index._plugins = smqtk.algorithms.relevancy_index._plugins",
                # Representations
                "smqtk.representation.classification_element._plugins"
                " = smqtk.representation.classification_element._plugins",
                "smqtk.representation.data_element._plugins = smqtk.representation.data_element._plugins",
                "smqtk.representation.data_set._plugins = smqtk.representation.data_set._plugins",
                "smqtk.representation.descriptor_element._plugins = smqtk.representation.descriptor_element._plugins",
                "smqtk.representation.descriptor_set._plugins = smqtk.representation.descriptor_set._plugins",
                "smqtk.representation.detection_element._plugins = smqtk.representation.detection_element._plugins",
                "smqtk.representation.key_value._plugins = smqtk.representation.key_value._plugins",
                # Web
                "smqtk.web._plugins = smqtk.web._plugins",
            ],
            'console_scripts': [
                'classifier_kfold_validation = smqtk.bin.classifier_kfold_validation:classifier_kfold_validation',
                'classifier_model_validation = smqtk.bin.classifier_model_validation:main',
                'smqtk-classify-files = smqtk.bin.classifyFiles:main ',
                'compute_classifications = smqtk.bin.compute_classifications:main',
                'compute_hash_codes = smqtk.bin.compute_hash_codes:main',
                'compute_many_descriptors = smqtk.bin.compute_many_descriptors:main',
                'smqtk-compute-descriptor = smqtk.bin.computeDescriptor:main',
                'smqtk-create-file-ingest = smqtk.bin.createFileIngest:main',
                'smqtk-create-girder-ingest = smqtk.bin.createGirderIngest:main',
                'descriptors_to_svmtrain = smqtk.bin.descriptors_to_svmtrainfile:main',
                'generate_image_transform = smqtk.bin.generate_image_transform:main',
                'iqr_app_model_generation = smqtk.bin.iqr_app_model_generation:main',
                'iqrTrainClassifier = smqtk.bin.iqrTrainClassifier:main',
                'make_balltree = smqtk.bin.make_balltree:main',
                'minibatch_kmeans_clusters = smqtk.bin.minibatch_kmeans_clusters:main',
                'smqtk-remove-old-files = smqtk.bin.removeOldFiles:main',
                'smqtk-proxy-manager-server = smqtk.bin.proxyManagerServer:main',
                'runApplication = smqtk.bin.runApplication:main',
                'smqtk-summarize-plugins = smqtk.bin.summarizePlugins:main',
                'train_itq = smqtk.bin.train_itq:main',
                'smqtk-make-train-test-sets = smqtk.bin.make_train_test_sets:main',
                'smqtk-nearest-neighbors = smqtk.bin.nearest_neighbors:main',
                'smqtk-check-images = smqtk.bin.check_images:main',
                'smqtk-nn-index-tool = smqtk.bin.nn_index_tool:cli_group',
            ]
        }
    )
