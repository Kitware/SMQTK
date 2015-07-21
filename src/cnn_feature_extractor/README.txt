
Using Caffe (http://caffe.berkeleyvision.org/), cnn_feature_extractor extracts a feature vector from each input image. It is assumed Caffe has been built and the header files and library (libcaffe.so) are available.

To run the program, first download two files
bvlc_reference_caffenet.caffemodel
imagenet_mean.binaryproto
from Caffe and copy them to
$SMQTK/data/caffenet

Then run the following command from the shell
cd $SMQTK_SRC

cnn_feature_extractor data/caffenet/bvlc_reference_caffenet.caffemodel data/caffenet/imagenet_val.prototxt fc7 data/caffenet/cnn 3 csv

cnn_feature_extractor data/caffenet/bvlc_reference_caffenet.caffemodel data/caffenet/imagenet_val.prototxt fc7 data/caffenet/cnn 3 svm

cnn_feature_extractor data/caffenet/bvlc_reference_caffenet.caffemodel data/caffenet/imagenet_val.prototxt fc7 data/caffenet/cnn 3 stdout >& /dev/null

