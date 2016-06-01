"""Example script for using Life"""
import numpy as np

import tensorflow as tf
import tensorflowhelper as tfh

def main():
    """Entry point function"""
    try:
        adder = tf.Variable(10.)
        adder2 = tf.Variable(7.)

        nn1 = tfh.NeuralNetwork(
            name="Inner-NN",
            layers=[
                tfh.ValidationLayer(name="Input Validation", shape=[None], dtype=tf.float32),
                tfh.OpLayer(lambda input: input + adder, [adder]),
                tfh.ValidationLayer(shape=[None], dtype=tf.float32)])
        nn2 = tfh.NeuralNetwork(
            name="Inner2-NN",
            layers=[
                tfh.ValidationLayer(name="Input Validation", shape=[None], dtype=tf.float32),
                tfh.OpLayer(lambda input: input + adder2, [adder2]),
                tfh.ValidationLayer(shape=[None], dtype=tf.float32)])

        life = tfh.Life(
            tfh.NeuralNetwork(
                name="Main-NN",
                layers=[nn1, nn2]),
            optimizer=tf.train.AdamOptimizer(0.3))

        input_x = np.array([
            0, 1, 2, 3, 4
        ], dtype=np.float32)

        expect_y = np.array([
            2, 3, 4, 5, 6
        ], dtype=np.float32)

        life.connect_neural_network(sample_input=input_x, sample_output=expect_y, will_train=True)

        life.init_var()
        # life.init_network([nn1, nn2])
        # life.load_saved_model("test.ckpt", nn2)

        # hypo = life.feed(input_x)

        # print(hypo)

        life.save_current_model("test.ckpt", nn1)

    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)

if __name__ == "__main__":
    main()
