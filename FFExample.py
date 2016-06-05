"""Example script for using FeedForwardLayer
Although it is recommended to use tfh.NeuralNetwork with tfh.Life
It is NOT required as you can see in this example.
"""
import tensorflow as tf
import tensorflowhelper as tfh

def main():
    """Entry point function"""
    sess = tf.Session()

    x_input = tf.placeholder(tf.float32, shape=[None])

    my_nn = tfh.NeuralNetwork(
        name="Main_Layer",
        layers=[
            tfh.ValidationLayer(shape=[None], dtype=tf.float32),
            tfh.ReshapeLayer(shape=[-1, 1]),
            tfh.FeedForwardLayer(features_out=2, name="FFLayer"),
            #features_out is the number of neuron in the layer
            tfh.FeedForwardLayer(features_in=2, features_out=5, name="FFLayer"),
            #features_in can also be set to verify
            tfh.ValidationLayer(shape=[None, 5], dtype=tf.float32),
        ]
    )

    y_hypothesis = my_nn.connect(x_input)

    sess.run(tf.initialize_all_variables())

    batch = [1, 2, 3, 4]

    print(sess.run(y_hypothesis, feed_dict={x_input: batch}))

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
