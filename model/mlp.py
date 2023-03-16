import tensorflow as tf
import tf_slim as slim
def MLP(x, name, reuse):
    assert x.shape.ndims in [2, 4] # (b, c) or (b, h, w, c)

    if x.shape.ndims == 4:
        x = tf.reduce_mean(input_tensor=x, axis=[1,2]) # (b, c)

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        x = slim.fully_connected(x, 512, tf.nn.relu)
        x = slim.fully_connected(x, 512, tf.nn.relu)
        x = slim.fully_connected(x, 512, tf.nn.relu)
    return x