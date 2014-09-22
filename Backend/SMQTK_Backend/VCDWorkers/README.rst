Video Descriptor Standardization Wrapping
=========================================

Since descriptors may be coming from subcontractors or as a already
compiled executable from some website, there is no standard input or output
to descriptors. Thus, we aim to create standardized wrapper objects for each
descriptor with the goal that they can then be plugged into a larger system.
As of now, the assumed standard input for a single descriptor execution is a
video file (\*.mp4) and the standard output is to a FeatureStore. It is up to
each descriptor's wrapper to know how to use the input video file, to execute
the descriptor in such a way that it produces valid output, and to store that
output into the FeatureStore. In the future, additional standardization
constraints may be levied, such as requiring descriptors to output video-level
feature vectors.

There are two parts to the currently implemented standardized run system:
The DescriptorWorkerInterface.py and the DescriptorWorkerRunner.py files. The
DescriptorWorkerInterface.py file defines a python class,
DescriptorWorkerInterface, to sub-class when creating a new descriptor wrapper
class. This interface class defines modules and parameters that must be
overridden to allow the new wrapper to plug in correctly to the running system.
The DescriptorWorkerRunner.py file is an executable script that easy allows
execution of any correctly implemented worker wrapper. This script is
essentially the front-end to generating descriptor features over many
videos.


Wrapping Example: Dense Trajectories Descriptor
===============================================
Currently, there is a single descriptor worker wrapper implemented and that
is for the Dense Trajectories descriptor:

    http://lear.inrialpes.fr/people/wang/dense_trajectories

This descriptor may be downloaded from the website above in source form. After
the descriptor has been built (see the source code for instructions), in order
to be able to use it with the DenseTrajectoriesDescriptor.py implementation,
the "DT_EXE_PATH" class variable must be modified (near the top of the class
definition) to be the full path to the executable just built.

This implementation shows the necessary components to override. In this
descriptor's case, there is an external executable to execute. It is also
specific to this descriptor that this executable output's its generated
features to standard out, so we can see that it captures that, and formats
the output into numpy arrays for storage into the FeatureStore.

CAVEAT: The current implementation, in its simplicity, relies on the most basic
construction signature for a FeatureStore object. Remember that by default, we
are using the SQLite implementation, which uses the current working directory
as the location for where it looks for and stores the SQLite database. If the
dense trajectories wrapper is to be run through the runner script, all runs
will need to be executed with the same directory as the CWD in order for all
runs to interact with the same data store. This will most likely be changed
in the future to be more intuitive.


Dense Trajectories Run Example
==============================

Let the following be an example command to run, using the runner script, to
execute a correctly implemented dense trajectories descriptor wrapper (assuming
that the executable path variable has already been changed at the top of the
class)::

    python DescriptorWorkerRunner.py -f {path-to}/DenseTrajectoriesDescriptor.py
                                     -i /path/to/video/file.mp4
                                     -v

Normally, the runner will set the logging level on the informational level. The
additional '-v' will output debugging messages also (warning and error messages
will always be output if encountered).