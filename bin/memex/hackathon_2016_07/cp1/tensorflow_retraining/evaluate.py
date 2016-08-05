"""
Preparing model:
 - Install bazel ( check tensorflow's github for more info )
    Ubuntu 14.04:
        - Requirements:
            sudo add-apt-repository ppa:webupd8team/java
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
        - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
          tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
        - chmod +x PATH_TO_INSTALL.SH
        - ./PATH_TO_INSTALL.SH --user
        - Place bazel onto path ( exact path to store shown in the output)
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb -- output_labels /path/output_labels.txt -- bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, change and run this code.
"""

import numpy as np
import tensorflow as tf
import os

imageDir = 'cp1eval/CP1_imageset'
# imageDir = 'subset'
# modelFullPath = '/tmp/output_graph.pb'
# labelsFullPath = '/tmp/output_labels.txt'
modelFullPath = 'run16_graph.pb'
labelsFullPath = 'run16_labels.txt'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images():
    answer = None

    # Creates graph from saved GraphDef.
    create_graph()

    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    print ','.join(['sha'] + labels)

    with tf.Session() as sess:
        for imageSubdir in os.listdir(imageDir):
            for imageFilename in os.listdir(os.path.join(imageDir, imageSubdir)):
                imagePath = os.path.join(imageDir, imageSubdir, imageFilename)

                if not tf.gfile.Exists(imagePath):
                    tf.logging.fatal('File does not exist %s', imagePath)
                    return answer

                image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
                print ','.join([os.path.join(imageSubdir, imageFilename)] + [str(p) for p in predictions])

                # top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
                # f = open(labelsFullPath, 'rb')
                # lines = f.readlines()
                # labels = [str(w).replace("\n", "") for w in lines]
                # for node_id in top_k:
                #     human_string = labels[node_id]
                #     score = predictions[node_id]
                #     print('%s (score = %.5f)' % (human_string, score))

                # answer = labels[top_k[0]]
                # return answer


if __name__ == '__main__':
    run_inference_on_images()
