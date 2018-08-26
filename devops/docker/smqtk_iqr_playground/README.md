SMQTK IQR Playground and Turn-key container
===========================================

We provide the docker container images:

    kitware/smqtk/iqr_playground_cpu
    kitware/smqtk/iqr_playground_nvidia

This is a self-contained SMQTK playground environment that can also act as an
image-to-application turn-key container. As an application, this container can
compute fresh models on mounted imagery using a default configuration setup,
using a custom configuration setup, or to run the IQR web-app with completely
custom models, database and configuration. Two ports are available for exposing
the IQR GUI web-application as well as the underlying IQR RESTful service
server.


### Quick Information

Default IQR web-app login:

    username: demo
    password: demo

This is modifiable via a JSON file located in the container:

    /usr/local/lib/python2.7/dist-packages/smqtk/web/search_app/modules/login/users.json

Container internal data directories for volume mounting:

    /images                     -- directory for automatic image discovery
    /home/smqtk/data/configs/   -- all configuration files
    /home/smqtk/data/db.mongo/  -- MongoDB database data files
    /home/smqtk/data/db.psql/   -- PostgreSQL database data files
    /home/smqtk/data/images/    -- symlink to /images
    /home/smqtk/data/logs/      -- all generated log and stamp files
    /home/smqtk/data/models/    -- common directory for model files


### Running IQR on New Imagery

One way to use this container is to treat it like an command line tool for
spinning up a new IQR ingest on a directory of images. This will pick up files
recursively in the mounted directory (uses command ``find <dir> -type f``):

    `docker run -d -v <abs-img-dir>:/images -p 5000:5000 kitware/smqtk/iqr_playground:...cpu... -b [-t]`

    OR

    `nvidia-docker run -d -v <abs-img-dir>:/images -p 5000:5000 kitware/smqtk/iqr_playground:...gpu... -b [-t]`

The use of ``nvidia-docker`` is required to use the GPU computation
capabilities (default options, can be changed and described later).
The ``-v`` option above shows where to mount a directory of images for
processing. The ``-p`` option above shows the default IQR web-app server port
that should be exposed.

The entrypoint script in this container can take a number of options:

    -h | --help
        Display the usage and options description.

    -b | --build
        Build model files for images mounted to /images

    -t | --tile
        Transform input images found in the images directory according to
        the provided generate_image_transform configuration JSON file.

When starting a new container, imagery must be mounted otherwise there will be
nothing to process/ingest. The ``-b`` option must also be given in order to
trigger model building.


## RESTful service

This container runs a RESTful service that provides the meat of the IQR
functionality. This is running on port 5001 inside the container and can be
published outside the container if desired.


## Runner Script

Included here are the bash scripts ``run_container.*.sh``. These are intended to
be a simple way of running the containers as is (i.e. with default
configurations) on a directory that [recursively] contains imagery to index
and perform IQR over.

This scripts may be called like as follows:

    $ run_container.cpu.sh /abs/path/to/image/dir [-t]

The above will run the container (CPU version in this case) as a daemon,
mounting the image directory and publishes the port 5000 (GUI) and 5001 (REST),
resulting in a running container named ``smqtk_iqr_cpu``.
The script then shows updating information about ongoing processing in the
container (literally a ``tail`` of certain log files inside the container).

The container and version used are defined by variables at the top of the
script, as well as what host port to publish to. By default these scripts try to
use the ``latest`` version of the docker images and will publish on ports 5000
and 5001.

When all the logs settle, mainly the ``runApp.IqrSearchDispatcher.log`` and the
``runApp.IqrService.log``, showing that the server has started, will the web
application be functional and interactive.


## Saving Generated Data

If models or other generated data from this container is to be saved in a more
permanent manner, the container should be started with more volume mounts than
just the input image directory in order for content to be saved to the host
filesystem instead of just within the container's filesystem.

Directories used in the container's filesystem:

- ``/home/smqtk/data/logs``
  - Default directory where log and stamp files are saved for commands
    processed. Stamp files mark what processes have finished successfully.

- ``/home/smqtk/data/models``
  - Generated model files are saved here by default.

- ``/home/smqtk/data/db.psql``
  - Directory where PostgreSQL database is generated if not mounted by the
    user.

- ``/home/smqtk/data/db.mongo``
  - Directory where MongoDB database is generated if not mounted by the user.

- ``/home/smqtk/data/image_tiles``
  - Directory where image tiles are saved if the ``-t`` or ``--tile``
    options are provided.


## Using Custom Configuration

While the default configuration files are fine for producing a generally usable
IQR instance, configurations may be extracted, modified and mounted to
``/home/smqtk/data/configs/``.

Extracting and modifying the default configuration files from the container is
probably the simplest way of customizing things. The following is a bash
snippet that will copy a ``configs`` directory containing the container's
default configs:

    $ docker run -dt --entrypoint bash --name ${CNAME} kitware/smqtk/iqr_playground:...cpu
    $ docker cp ${CNAME}:/home/smqtk/data/configs/ ${OUTPUT_DIR}
    $ docker stop ${CNAME}
    $ docker rm ${CNAME}

To use the custom configuration files, simply mount the containing directory to
``/home/smqtk/data/configs`` when running the container.

**Note:** *When mounting directory of configuration files, it must contain all
configuration files that were extracted as this is the only place configuration
files are located in the container. If the entrypoint configuration was
modified, then files may be named other than their default names. Only the
``entrypoint.conf`` file cannot be renamed (assumed by entrypoint script).*

Configuration files and what they are used for:

- ``entrypoint.conf``
  - Defines container entrypoint script variables, like directories to use
    within ``/home/smqtk/data/``, the names of configuration files for the
    different tools used, and command line parameters for tools that take
    them.

- ``psql_table_init.sql``
  - Internal PostgreSQL database table initialization for image descriptor
    storage. If descriptors are to be stored in a different way, this file
    may be empty.

- ``generate_image_transform.tiles.json``
  - Configuration file for optional image tile generation. Importantly,
    this controls the size/shape of the extracted tiles.

- ``compute_many_descriptors.json``
  - Configuration file for utility that computed image content descriptors.

- ``train_itq.json``
  - Configuration file for utility that trains ITQ locality-sensitive hash
    functor models.

- ``compute_hash_codes.json``
  - Configuration file for utility that computed LSH codes for indexed
    imagery.

- ``make_balltree.json``
  - Configuration for the utility that generates a ball-tree index of LSH hash
    codes generated in support of the LSH algorithm.

- ``runApp.IqrSearchDispatcher.json``
  - Configuration file for SMQTK IQR search web application. It is wise
    to change the ``SECRET_KEY`` option in here if the application is to
    be publicly faced.

- ``runApp.IqrService.json``
  - Configuration file for the SMQTK IQR RESTful service. It is wise
    to change the ``SECRET_KEY`` option in here if the application is to
    be publicly faced.


### Troubleshooting

Q: My container quickly stopped after I executed the above "docker run..."
command.

    Re-run the ``docker run...`` with an additional volume mount to save out
    the log files: ``-v <some-local-dir>:/home/smqtk/data/logs``. A more
    descriptive error message should be present in the log for the failing
    utility (grep -i for "error" on log files).

TODO: More error situations. If a confusing situation arises, email
[paul.tunison@kitware.com](mailto:paul.tunison@kitware.com) and we can add 
new Q&As here!
