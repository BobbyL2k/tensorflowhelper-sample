import tensorflow as tf
import tensorflowhelper as tfh

"""
Although it is recommended to use tfh.NeuralNetwork with tfh.Life
It is NOT required as you can see in this example.
"""

try:
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None])
    y_ = tf.placeholder(tf.float32, shape=[None])

    print(x._shape_as_list())

    W = tf.Variable(2.)
    b = tf.Variable(3.)

    myNN = tfh.NeuralNetwork(
        name = "Main Layer",
        layers = [
            tfh.ValidationLayer( shape=[None], dtype=tf.float32 ),
            tfh.ReshapeLayer( shape=[-1,1] ),
            tfh.FeedForwardLayer( features_out=2, name="FFLayer" ), #features_out is the number of neuron in the layer
            tfh.FeedForwardLayer( features_in=2, features_out=5, name="FFLayer" ), #features_in can also be set to verify
            tfh.ValidationLayer( shape=[None, 5], dtype=tf.float32 ),
        ]
    )

    y = myNN.connect(x)

    sess.run(tf.initialize_all_variables())

    batch = [1,2,3,4];

    print(sess.run(y, feed_dict={x: batch}))
    
except tfh.utilities.TFHError as tfhError:
    print(tfhError)
