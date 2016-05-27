import tensorflow as tf
import tensorflowhelper as tfh

try:
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None])
    y_ = tf.placeholder(tf.float32, shape=[None])

    print(x._shape_as_list())

    W = tf.Variable(2.)
    b = tf.Variable(3.)

    myNN = tfh.NeuralNetwork(
        layers = [
            tfh.ValidationLayer( shape=[None], dtype=tf.float32 ),
            tfh.ReshapeLayer( shape=[-1,1] ),
            tfh.FeedForwardLayer( features_out=2, name="FFLayer" ),
            tfh.ValidationLayer( shape=[None, 2], dtype=tf.float32 ),
        ],
        name = "Main Layer"
    )

    y = myNN.connect(x)

    sess.run(tf.initialize_all_variables())

    batch = [1,2,3,4];

    print(sess.run(y, feed_dict={x: batch}))
    
except tfh.utilities.TFHError as tfhError:
    print(tfhError)
