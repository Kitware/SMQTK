LEEDS Butterfly IQR Service Use Case
====================================
In this document we provide a complete interaction with this web service.
This example use case was created by running the
``kitware/smqtk/iqr_playground`` docker image over the `LEEDS Butterfly
dataset`_.
Other versions of this image should provide similar results.

The steps we will show here include starting the image over the target dataset,
starting an IQR session with the UID of a known image in the data set and
iterating through a couple refinement stages.

Starting the image on the dataset
---------------------------------
First, we must start the services over the target dataset.
In our example here, this is the LEEDS Butterfly dataset as provided by the
`LEEDS Butterfly dataset`_ download::

    $ docker run --rm -v $PWD/images:/home/smqtk/data/images -p \
    5000-5001:5000-5001 --name smqtk_iqr_leeds \
    kitware/smqtk/iqr_playground:latest -b

IQR Process
-----------
For interacting with the service we will use the curl command syntax below.
After each curl command we will show the JSON return that is expected.

Initialize a session. In this example, we will manually set the UID to be 0 for
the sake of consistency::

    $ curl -X POST localhost:5001/session -d sid=0
    {
      "message": "Created new session with ID '0'",
      "sid": "0",
      "time": {
        "unix": 1483650560.699204,
        "utc": "Thu Jan  5 21:09:20 2017"
      }
    }

We pick an image of one of the monarch butterflies from the dataset:
``001_0001.jpg``.
The SHA1 checksum of this file, which is used as the UID of the data and its
descriptor in the service: ``84f62ef716fb73586231016ec64cfeed82305bba``.
In the following call, the value given to the ``pos_uuids`` parameter should be
a JSON formatted list of string UID values::

    $ curl -X
    $ curl -X PUT localhost:5001/refine -d sid=0 \
      -d 'pos_uuids=["84f62ef716fb73586231016ec64cfeed82305bba"]'
    {
      "message": "Refine complete",
      "sid": "0",
      "time": {
        "unix": 1483650831.972214,
        "utc": "Thu Jan  5 21:13:51 2017"
      }
    }

Let's take a look at the results after the first refine.
Since there are many images in the `LEEDS Butterfly dataset`_, we will only
query for the top 5  here for convenience.
In the ``results`` list, each sub-list is an image  descriptor UID (SHA1
checksum) and confidence pair.
Confidence values are non-deterministic in value, but deterministic in relative
order.
For example, the confidence score for UID
``ad4af38cf36467f46a3d698c1720f927ff729ed7`` might not be ``0.7059437607144408``
every time, but it will always be second given this specific example's chain of
events::

    $ curl -X GET localhost:5001/get_results?sid=0
    {
      "i": 0,
      "j": 5,
      "message": "Returning result pairs",
      "results": [
        [
          "84f62ef716fb73586231016ec64cfeed82305bba",
          1.0
        ],
        [
          "ad4af38cf36467f46a3d698c1720f927ff729ed7",
          0.7059437607144408
        ],
        [
          "c3b612d0e5f1014502393a3efe81293137d6bc0b",
          0.703764046397459
        ],
        [
          "eb4a2c97fdb4fde289aa297bee70fb5813137670",
          0.6970641513036197
        ],
        [
          "0c6af65759c958eb21aa83a885367504b601a787",
          0.696986304178252
        ]
      ],
      "sid": "0",
      "time": {
        "unix": 1483650955.393136,
        "utc": "Thu Jan  5 21:15:55 2017"
      },
      "total_results": 500
    }

It turns out the that images with UID
``ad4af38cf36467f46a3d698c1720f927ff729ed7`` (001_0032.jpg) and
``c3b612d0e5f1014502393a3efe81293137d6bc0b`` (001_0058.jpg) are also monarch
butterflies.
Let us perform a second refinement marking those as positive as
well::

    $ curl -X PUT localhost:5001/refine -d sid=0 \
      -d 'pos_uuids=["84f62ef716fb73586231016ec64cfeed82305bba",
                     "ad4af38cf36467f46a3d698c1720f927ff729ed7",
                     "c3b612d0e5f1014502393a3efe81293137d6bc0b"]'
    {
      "message": "Refine complete",
      "sid": "0",
      "time": {
        "unix": 1483651728.440528,
        "utc": "Thu Jan  5 21:28:48 2017"
      }
    }

Getting the new results::

    $ curl -X GET localhost:5001/get_results?sid=0
    {
      "i": 0,
      "j": 5,
      "message": "Returning result pairs",
      "results": [
        [
          "ad4af38cf36467f46a3d698c1720f927ff729ed7",
          1.0
        ],
        [
          "84f62ef716fb73586231016ec64cfeed82305bba",
          1.0
        ],
        [
          "c3b612d0e5f1014502393a3efe81293137d6bc0b",
          1.0
        ],
        [
          "eb4a2c97fdb4fde289aa297bee70fb5813137670",
          0.9999999997837792
        ],
        [
          "e8627a1a3a5a55727fe76848ba980c989bcef103",
          0.9999999996915796
        ]
      ],
      "sid": "0",
      "time": {
        "unix": 1483651935.69592,
        "utc": "Thu Jan  5 21:32:15 2017"
      },
      "total_results": 573
    }

We now see that the next two results after our initial query and two
adjudications, which are in truth monarch examples (images ``001_0070.jpg`` and
``001_0025.jpg`` respectively), show a much higher confidence.

Updated Process
---------------

New command flow since API overhaul (2017, Nov-Dec)::

    # Initialize a new session with custom ID 0.
    curl -X POST localhost:5001/session -d sid=0

    # Positively adjudicate first monarch image (001_0001.jpg)
    curl -X POST localhost:5001/adjudicate -d sid=0 \
        -d 'pos=["84f62ef716fb73586231016ec64cfeed82305bba"]'

    # Negatively adjudicate first image of every other type (0[02-10]_0001.jpg)
    curl -X POST localhost:5001/adjudicate -d sid=0 \
        -d 'neg=["2e56b2dda41c2bd734b789d8b4c1a4b253cdb414",
                 "17bd8bbee8addc5f7f4b00b222824de240cea5eb",
                 "a35586f14c2cb3571f8cd836ed394febc9f28a67",
                 "8c5f0de4416ceefbc2281c901218c134868502dd",
                 "f5fff08ba1a9340905434011688966e721f2463b",
                 "b2d0b1d9c72ff32304bde06daac7f6c69fcd26c4",
                 "02e057a67b5fe8705c2d95309333ca9bbea6d948",
                 "01ad8b533b564c75e12180f88f9138e97e24ed7e",
                 "f56f43ab5eda0fc58c46d22b299868a7b4b39dee"]'

    # Initialize working index
    curl -X POST localhost:5001/initialize -d sid=0

    # Refine with no custom PR-bias.
    curl -X POST localhost:5001/refine -d sid=0 -d pr_adj=0.0
    #curl -X POST localhost:5001/refine -d sid=0 -d pr_adj=-10.0
    #curl -X POST localhost:5001/refine -d sid=0 -d pr_adj=10.0

    # Get results for confirmation of similarity.
    curl -sGX GET localhost:5001/get_results -d sid=0

.. _LEEDS Butterfly dataset: http://www.josiahwang.com/dataset/leedsbutterfly/
