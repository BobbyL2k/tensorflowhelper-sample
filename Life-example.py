import numpy as np

import tensorflow as tf
import tensorflowhelper as tfh

try:
    adder = tf.Variable(10.)

    life = tfh.Life(
            tfh.NeuralNetwork( name = "Inner-NN", layers = [
                tfh.ValidationLayer( name="Input Validation", shape=[None], dtype=tf.float32 ),
                tfh.OpLayer(lambda input: input + adder),
                tfh.ValidationLayer( shape=[None], dtype=tf.float32 )
            ] ), optimizer=tf.train.AdamOptimizer(0.3) );
    
    input_x = np.array([
        1,2,3,4
    ], dtype=np.float32)
    
    expect_y = np.array([
        3,4,5,6
    ], dtype=np.float32)
    
    life.connectNeuralNetwork( sampleInput=input_x, sampleOutput=expect_y, willTrain=False )
    
    life.initVar()

    hypo = life.feed(input_x);
    
    print(hypo)
    
    for c in range(200):
        cost = life.train(input_x, expect_y, processList=[adder])
        if c%10 == 0:
            print(cost)
        
    
except tfh.utilities.TFHError as tfhError:
    print(tfhError)
