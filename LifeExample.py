"""Example script for using Life"""
import numpy as np

import tensorflow as tf
import tensorflowhelper as tfh

def main():
    """Entry point function"""
    adder = tf.Variable(10.)

    life = tfh.Life(
        tfh.NeuralNetwork(
            name="Inner_NN",
            layers=[
                # Even though for ValidationLayer's name isn't required to not include space
                # but it is good to stay consistance since other might require it
                tfh.ValidationLayer(name="Input_Validation", shape=[None], dtype=tf.float32),
                tfh.OpLayer(lambda input: input + adder, [adder]),
                tfh.ValidationLayer(shape=[None], dtype=tf.float32)]),
        optimizer=tf.train.AdamOptimizer(0.3))

    input_x = np.array([
        1, 2, 3, 4
    ], dtype=np.float32)

    expect_y = np.array([
        3, 4, 5, 6
    ], dtype=np.float32)

    life.connect_neural_network(sample_input=input_x, sample_output=expect_y, will_train=True)

    life.init_var()

    hypo = life.feed(input_x)

    print(hypo)

    for counter in range(200):
        cost = life.train(input_x, expect_y, process_list=[adder])
        if counter%10 == 0:
            print(cost)

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
