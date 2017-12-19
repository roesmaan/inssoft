from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np

# Supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(argc, argv):
    # Declare dataset file names
    trainfile = "trainvalues.csv"
    testfile = "testvalues.csv"

    # Import dataset and create a classifier
    traindata = tf.contrib.learn.datasets.base.load_csv_without_header(filename=trainfile, target_dtype=np.int, features_dtype=np.float32)
    testdata = tf.contrib.learn.datasets.base.load_csv_without_header(filename=testfile, target_dtype=np.int, features_dtype=np.float32)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[13])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[130, 78, 13], n_classes=2, model_dir=os.getcwd()+"/diabetes_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(traindata.data)}, y=np.array(traindata.target), num_epochs=None, shuffle=True)

    # Train model
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Test model
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(testdata.data)}, y=np.array(testdata.target), num_epochs=1, shuffle=False)

    # Determine accuracy of the model
    score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("Test accuracy: {0:f}\n".format(score))

    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
