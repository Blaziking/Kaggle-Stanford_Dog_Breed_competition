import tensorflow as tf
import numpy as np
import dataset_loader
import os

training_size = 9000
depth = 120 #number of dog categories
minibatch_size = 256 
num_features =196608  #256*256*3=196608
n_h =[200,150,120]  #number of hidden units
epochs = 1000
learning_rate=0.0001


dataset = dataset_loader.load_data(training_size)
training_dataset = dataset['training_dataset']
validation_dataset = dataset['validation_dataset']

home_dir = os.path.dirname(os.path.realpath('__file__'))

def initialize_variables():
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        #W1 is filter for first cnn layer. shape is (height,width,in_channel,out_channel)
        W1 = tf.get_variable('W1',shape=[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable('W2',shape=[6,6,8,16],initializer= tf.contrib.layers.xavier_initializer())
        parameters = {'W1':W1, 'W2':W2}
    return parameters


def forward_prop(X,parameters):
    '''Arguments:
               X of shape (num_featues,batch_size)
               parameters-- dictionary of parameters


      Returns:
      logits-- without calculating actiavation function on the last layer as the cost function doesn't need it. 
    '''
    W1 = parameters['W1']
    W2 = parameters['W2']
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='VALID') 
    # Z2 = tf.nn.conv2d(P1,W2, strides=[1,1,1,1],padding='VALID')
    # A2 = tf.nn.relu(Z2)
    # P2 = tf.nn.max_pool(A2,ksize=[1,8,8,1],strides=[1,8,8,1],padding='VALID') 
    P2 = tf.contrib.layers.flatten(P1)
    logits = tf.contrib.layers.fully_connected(P2,120,activation_fn=None)
    return logits


def compute_cost(Z,Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels = labels))
    return cost

def model(training_dataset,validation_dataset,learning_rate=0.00002,epochs=10):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        batched_training_dataset = training_dataset.batch(minibatch_size)
        iterator = tf.data.Iterator.from_structure(batched_training_dataset.output_types,batched_training_dataset.output_shapes)
        

        training_init_op = iterator.make_initializer(batched_training_dataset)
#        validation_init_op = iterator.make_initializer(batched_validation_dataset)
        next_element = iterator.get_next()
        data = next_element     
        X = data[0]
        Y = data[1]
#        assert(X.get_shape == (minibatch_size))
        # assert(Y.shape == (depth,minibatch_size))

        parameters = initialize_variables()
        logits =  forward_prop(X,parameters)    
        loss = compute_cost(logits,Y)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        predictions = tf.reshape(tf.argmax(logits,axis=0),shape=[-1,1])
        labels = tf.reshape(tf.argmax(Y,axis=0), shape=[-1,1])
        accuracy_calculator,accuracy_updater = tf.metrics.accuracy(predictions=predictions,labels=labels,name='metric')
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope = 'metric')
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        

        saver = tf.train.Saver(parameters, max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in range(epochs):
                sess.run(running_vars_initializer)
                epoch_cost = 0 
                num_minibatches  = int(training_size/minibatch_size)
                sess.run(training_init_op)
                while True:
                    try:
                        _, cost = sess.run([optimizer,loss])
                        sess.run(accuracy_updater)
                        epoch_cost = epoch_cost + cost
                    except tf.errors.OutOfRangeError:
                        if epoch%10 == 0:
                            accuracy = sess.run(accuracy_calculator)

                            epoch_cost = epoch_cost/num_minibatches
                            print('cost is {0} and accuracy is {2} for epoch {1}'.format(epoch_cost,epoch,accuracy))
                        break
                
            print("Final accuracy " + str(sess.run(accuracy_calculator)))

            saver.save(sess, home_dir+'/model_parameters')
         
model(training_dataset,validation_dataset,learning_rate=learning_rate,epochs=epochs)
print('works')
