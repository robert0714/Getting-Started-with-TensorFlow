#using_tensorboard.py

import tensorflow as tf

a = tf.constant(10,name="a")
b = tf.constant(90,name="b")
y = tf.Variable(a+b*2,name='y')
model = tf.global_variables_initializer()

with tf.Session() as session:
    # merged = tf.merge_all_summaries()
    merged = tf.summary.merge_all()
    # writer = tf.train.SummaryWriter("/tmp/tensorflowlogs",session.graph)
    writer = tf.summary.FileWriter("/tmp/tensorflowlogs",session.graph)
    session.run(model)
    print(session.run(y))


