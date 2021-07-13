Steps of the SMQTK Release Process
==================================
Three types of releases are expected to occur:
- major
- minor
- patch

See the ``CONTRIBUTING.md`` file for information on how to contribute features
and patches.

The following process should apply when any release that changes the version
number occurs.

Create and merge version update branch
--------------------------------------

Patch Release
^^^^^^^^^^^^^
A patch release should only contain fixes for bugs or issues with an existing
release.
No new features or functionality should be introduced in a patch release.
As such, patch releases should only ever be based on an existing release point.

1. Create a new branch off of the ``release`` branch named something like
   ``release-patch-{NEW_VERSION}``.

  - Increment patch value in ``python/smqtk/__init__.py`` file's
    ``__version__`` attribute.
  - Rename the ``docs/release_notes/pending_patch.rst`` file to
    ``docs/release_notes/v{VERSION}.rst``, matching the value in the
    ``__version__`` attribute.  Add a descriptive paragraph under the title
    section summarizing this release.
  - Add new release notes RST file reference to ``docs/release_notes.rst``.

2. Tag branch (see `Tag new version`_ below ).
3. Merge version bump branch into ``release`` and ``master`` branches.

Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^
Major and minor releases may add one or more trivial or non-trivial features
and functionalities.

1. Create a new branch off of the ``master`` or ``release`` named something
   like ``release-[major,minor]-{NEW_VERSION}``.

  a) Increment patch value in ``VERSION`` file.
  b) Rename the ``docs/release_notes/pending_release.rst`` file to
     ``docs/release_notes/v{VERSION}.rst``, matching the value in the
     ``VERSION`` file.  Add a descriptive paragraph under the title section
     summarizing this release.
  c) Add new release notes RST file reference to ``docs/release_notes.rst``.

2. Create a pull/merge request for this branch with master as the merge target.
   This is to ensure that everything passes CI testing before making the
   release. If there is an issue then branches should be made and merged into
   this branch until the issue is resolved.
3. Tag branch (see `Tag new version`_ below) after resolving issues and before
   merging into ``master``.
4. Reset the release branch (--hard) to point to the new branch/tag.
5. Merge version bump branch into the ``master`` branch.

Tag new version
---------------
Release branches should be tagged in order to record where in the git tree a
particular release refers to.
The branch off of ``master`` or ``release`` is usually the target of such tags.

Currently the ``From GitHub`` method is preferred as it creates a "verified"
release.

From GitHub
^^^^^^^^^^^
Navigate to the `releases page on GitHub`_ and click the ``Draft a new
release`` button in upper right.

Fill in the new version in the ``Tag version`` text box (e.g. ``v#.#.#``)
and use the same string in the ``Release title`` text box.
The "@" target should be the release branch created above.

Copy and past this version's release notes into the ``Describe this release``
text box.

Remember to check the ``This is a pre-release`` check-box if appropriate.

Click the ``Public release`` button at the bottom of the page when complete.

From Git on the Command Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a new git tag using the new version number (format:
``v<MAJOR.<MINOR>.<PATCH>``) on the merge commit for the version update branch
merger::
    $ git tag -a -m "[Major|Minor|Patch] release v#.#.#"

Push this new tag to GitHub (assuming origin remote points to `SMQTK on
GitHub`_::
    $ git push origin v#.#.#

To add the release notes to GitHub, navigate to the `tags page on GitHub`_
and click on the "Add release notes" link for the new release tag.  Copy and
paste this version's release notes into the description field and the version
number should be used as the release title.

Create new version release to PYPI
----------------------------------
Make sure the source is checked out on the newest version tag, the repo is
clean (no uncommited files/edits), and the ``build`` and ``dist`` directories
are removed::
    $ git checkout <VERSION_TAG>
    $ rm -r dist python/smqtk.egg-info

Create the ``build`` and ``dist`` files for the current version with the
following command(s) from the source tree root directory::
    $ python setup.py sdist

Make sure your ``$HOME/.pypirc`` file is up-to-date and includes the following
section with your username/password::
    [pypi]
    username = <username>
    password = <password>

Make sure the ``twine`` python package is installed and is up-to-date and then
upload dist packages created with::
    $ twine upload dist/*


.. _SMQTK on GitHub: https://github.com/Kitware/SMQTK
.. _releases page on GitHub: https://github.com/Kitware/SMQTK/releases
.. _tags page on GitHub: https://github.com/Kitware/SMQTK/tags
