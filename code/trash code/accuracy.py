import tensorflow as tf
import numpy as np

labels = (np.array(([0,1,1,1],[1,0,0,0],[0,0,0,0],[0,0,0,0])))
logits = (np.array(([0,4,0.9,0.1],[0.3,0.8,0.1,0.3],[0.3,0.7,0.1,0.2],[0.4,0.6,0.1,0.2])))

labels = tf.argmax(labels,0)
predictions = tf.argmax(logits,0)

accuracy_calculator,accuracy_updater = tf.metrics.accuracy(predictions=predictions,labels=labels,name='metric')
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope = 'metric')
running_vars_initializer = tf.variables_initializer(var_list=running_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(running_vars_initializer)
    sess.run(accuracy_updater)
    print('accuracy is :  ' + str(sess.run(accuracy_calculator)) )
