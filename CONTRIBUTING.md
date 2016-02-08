# Contributing to SMQTK

Here we describe at a high level how to contribute to SMQTK.
See the [SMQTK README] file for additional information.


## The General Process

1.  The official SMQTK source is maintained [on GitHub]

2.  Fork SMQTK into your user's namespace [on GitHub].

3.  Create a local clone of the SMQTK repository to work on.

        $ git clone https://github.com/Kitware/SMQTK.git smqtk
        $ cd smqtk
        
    The main repository will be called ``origin`` by default.

4.  Add the URL of your user namespace fork of SMQTK as a remote.

        $ git remote add <username-or-label> https://github.com/<username>/SMQTK.git

5.  Build SMQTK and source the setup script.
    See the [build instructions] for more details.

6.  Create a topic branch, edit files and create commits:

        $ git checkout -b <branch-name>
        $ <edit things>
        $ git add <file1> <file2> ...
        $ git commit

7.  Push topic branch with commits to your fork in GitHub:

        $ git push <username-or-label> HEAD -u

8.  Visit the Kitware SMQTK page [on GitHub], browse to the "Pull requests" tab
    and click on the "New pull request" button in the upper-right.
    Click on the "compare across forks" link, browse to your fork and browse to
    the topic branch to submit for the pull request.
    Finally, click the "Create pull request" button to create the request.


SMQTK uses GitHub for code review and [Travis-CI] for continuous testing as new
pull requests are made.
All checks/tests must pass before a PR can be merged.

Sphinx is used for manual and automatic API [documentation]. 


[SMQTK README]: README.md
[on GitHub]: https://github.com/Kitware/SMQTK
[build instructions]: http://smqtk.readthedocs.org/en/latest/building.html
[Travis-CI]: https://travis-ci.org/Kitware/SMQTK/
[documentation]: docs/
