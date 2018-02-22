import tensorflow as tf
from  os.path import abspath

w = tf.Variable(0,dtype=tf.float32)
cost = w**2 + 20*w +100
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

summary = tf.summary.image(tensor=w,name='w')

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    #summary_writer = tf.summary.FileWriter(str(abspath),session.graph_def)
    for i in range(30):
        _,loss = session.run([train,cost])
        if i%2 == 0:
            print('cost for {0}th iteration  is {1}'.format(i,loss)) 
        #summary_writer.add_summary(summary,i)
    print(session.run(w))
