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
                
        W1 = tf.get_variable(name='W1',shape=(n_h[0], num_features),initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        W2 = tf.get_variable(name='W2',shape=(n_h[1], n_h[0]),initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        W3 = tf.get_variable(name='W3',shape=(n_h[2], n_h[1]),initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        
        W4 = tf.get_variable(name='W4',shape=(depth,n_h[2]),initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

        b1 = tf.get_variable(name='b1',shape=[n_h[0],1],initializer=tf.zeros_initializer(),dtype=tf.float32)
        b2 = tf.get_variable(name='b2',shape=[n_h[1],1],initializer=tf.zeros_initializer(),dtype=tf.float32)
        b3 = tf.get_variable(name='b3',shape=[n_h[2],1],initializer=tf.zeros_initializer(),dtype=tf.float32)        
        b4= tf.get_variable(name='b4',shape=[depth,1],initializer=tf.zeros_initializer(),dtype=tf.float32) 
        parameters = {'b1':b1,'b2':b2,'b3':b3,'b4':b4,'W1':W1, 'W2':W2, 'W3':W3, 'W4':W4}
    return parameters


def forward_prop(X,parameters):
    '''Arguments:
               X of shape (num_featues,batch_size)
               parameters-- dictionary of parameters


      Returns:
      logits-- without calculating actiavation function on the last layer as the cost function doesn't need it. 
    '''
        
    Z1 = tf.matmul(parameters['W1'],X)+ parameters['b1']
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(parameters['W2'],A1)+ parameters['b2']
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(parameters['W3'],A2)+ parameters['b3']
    A3 = tf.nn.relu(Z3)
    logits = tf.matmul(parameters['W4'],A3)+parameters['b4']
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
        X = tf.transpose(data[0])
        Y = tf.transpose(data[1])
        # assert(X.shape == (num_features,minibatch_size))
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
