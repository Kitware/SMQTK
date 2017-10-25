This directory contains a Jupyter notebook for testing the classifier service, along with some associated files:

- classifier_test.ipynb: You can run this notebook to test out the server.
- dummy_classifier.py, dummy_descriptor_generator.py: Files with dummy implementations of necessary classes.
- dummy_classifier.pkl.b64: a base64-encoded pickle file of an instance of DummyClassifier, for testing upload
- fish-bike.jpg.b64: a base64-encoded JPG file for use as input to classify
- test_classifier_server.sh: A script that runs several curl commands, to test the output of the server. Run this after starting the server in the notebook.
- test_classifier_responses.txt: A text file containing the output of a run of `test_classifier_server.sh`