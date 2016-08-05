
# TensorFlow Retraining

For the CP1 classification task, I used the instructions in TensorFlow's
[How to Retrain Inception's Final Layer for New Categories](https://www.tensorflow.org/versions/r0.10/how_tos/image_retraining/index.html)
how-to article. Some notes:

* I split the full data into 50% train, 25% validate, 25% test data using
the `split.py` script in this directory. The inputs were Paul Tunison's
4-way split of the data that were ensured to be split by cluster. The splitting
explicitly prefixes the image files with `train_`, `validate_`, and `test_`
in `pos` and `neg` directories, to signal to TensorFlow which images to use
for each purpose. There is a `size` variable in `split.py` which can be
adjusted to get a subset of the data, or you can simply pull the full data
into the `pos` and `neg` directories.

* The TensorFlow `retrain.py` script was edited (edited version here) in
order to pick up the train, validate, and test images explicitly instead of
randomly.

* The command to run the experiments (note that I replaced the `retrain.py`
script in-place in the `tensorflow/examples/image_retraining/` source directory):

  ```
  time bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir dir/of/split/command --how_many_training_steps 1000 \
    --learning_rate 0.01 --train_batch_size 5000 --test_batch_size 10000 \
    --validation_batch_size 1000
  ```

  Since the data was so noisy, increasing the batch size seemed to
increase the performance of the training.

* After the model is trained, the resulting model and labels are placed in `/tmp/output_graph.pb` and `/tmp/output_labels.txt`. The `evaluate.py` script
will produce classification results on a set of evaluation data (another
directory of images). Note that for this script you need TensorFlow installed
locally, which I did from source by following
[these instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#installing-from-sources).

* Finally, the image scores need to be rolled up into cluster scores. The
`aggregate.py` will roll up the scores by taking the average or maximum of
image scores and output a JSON lines file that was submitted (after
manually including some extra clusters that we had no data for).

Some helpful links:

* http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/

* http://axon.cs.byu.edu/papers/Wilson.nn03.batch.pdf
