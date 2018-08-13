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
1. Create a new branch off of the ``release`` branch named something like
   ``release-patch-{NEW_VERSION}``.
    - Increment patch value in ``VERSION`` file.
    - Rename the ``docs/release_notes/pending_patch.rst`` file to
      ``docs/release_notes/v{VERSION}.rst``, matching the value in the
      ``VERSION`` file.
    - Add new release notes RST file reference to ``docs/release_notes.rst``.
2. Merge version bump branch into ``release`` and ``master`` branches.

Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^
1. Create a new branch off of the ``master`` branch named something like
   ``release-[major,minor]-{NEW_VERSION}``.
    - Increment patch value in ``VERSION`` file.
    - Rename the ``docs/release_notes/pending_release.rst`` file to
      ``docs/release_notes/v{VERSION}.rst``, matching the value in the
      ``VERSION`` file.
    - Add new release notes RST file reference to ``docs/release_notes.rst``.
2. Merge version bump branch into the ``master`` branch.
3. Reset the release branch (--hard) to point to the new master.

Tag new version
---------------
Create a new git tag using the new version number (format:
``v<MAJOR.<MINOR>.<PATCH>``) on the merge commit for the version update branch
merger::
    $ git tag -a -m "[Major|Minor|Patch]" release v#.#.#

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
    $ git check <VERSION_TAG>
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
.. _tags page on GitHub: https://github.com/Kitware/SMQTK/tags
