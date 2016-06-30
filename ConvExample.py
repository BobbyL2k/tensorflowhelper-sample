"""Example script for using ConvNet"""
import numpy as np

import tensorflow as tf
import tensorflowhelper as tfh

def main():
    """Entry point function"""
    life = tfh.Life(
        tfh.NeuralNetwork(
            name="Inner_NN",
            layers=[
                tfh.ValidationLayer(name="Input_Validation",
                                    shape=[None, 10, 10], dtype=tf.float32),
                tfh.ReshapeLayer(shape=[None, 10, 10, 1]),
                tfh.ConvLayer(depth_out=2, kernel_width=2),
                tfh.ConvBiasLayer(),
                tfh.ConvLayer(depth_out=1, kernel_width=3, padding=False),
                tfh.ConvBiasLayer(),
                tfh.ValidationLayer(shape=[None, 8, 8, 1], dtype=tf.float32),
                tfh.ReshapeLayer(shape=[None, 8, 8])
                ]),
        optimizer=tf.train.AdamOptimizer(0.3))

    input_x = np.random.randn(1, 10, 10).astype(np.float32)

    expect_y = np.random.randn(1, 8, 8).astype(np.float32)

    life.connect_neural_network(sample_input=input_x, sample_output=expect_y, will_train=True)

    life.init_var()

    hypo = life.feed(input_x)

    print(hypo)

    for counter in range(200):
        cost = life.train(input_x, expect_y, process_list=[])
        if counter%10 == 0:
            print(cost)

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
