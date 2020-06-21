#get the trained model embedding
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
file_name ='/home/ddd/data/cnndailymail3/finished_files/exp_logs/train/model.ckpt-33831'#model name
name_variable_to_restore='seq2seq/embedding/embedding'#variable name
reader = pywrap_tensorflow.NewCheckpointReader(file_name)#read .ckpt
var_to_shape_map = reader.get_variable_to_shape_map()#get variable
print('shape', var_to_shape_map[name_variable_to_restore]) #print shape is 'shape [50000, 128]'
my_embedding = tf.get_variable("my_embedding", var_to_shape_map[name_variable_to_restore], trainable=False)#rename 'embedding'variable name
s1=tf.nn.embedding_lookup(my_embedding,[1,3])#id=1 and id=3 embedding is 1*128,get what I need
sess = tf.Session()   #create seesion
sess.run(tf.variables_initializer([my_embedding], name='init'))#run variable
print(sess.run(s1))#print the s1 value
print(sess.run(my_embedding))#print the my_embedding value
print(my_embedding)
print(type(my_embedding))

"""a small example 
p=tf.Variable(tf.random_normal([10,1]))#生成10*1的张量
b = tf.nn.embedding_lookup(p, [1, 3])#查找张量中的序号为1和3的
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    #print(c)
    print(sess.run(p))
    print(p)
    print(type(p))
"""
"""output
[[0.15791859]
 [0.6468804 ]]
[[-0.2737084 ]
 [ 0.15791859]
 [-0.01315552]
 [ 0.6468804 ]
 [-1.4090979 ]
 [ 2.1583703 ]
 [ 1.4137447 ]
 [ 0.20688428]
 [-0.32815856]
 [-1.0601649 ]]
<tf.Variable 'Variable:0' shape=(10, 1) dtype=float32_ref>
<class 'tensorflow.python.ops.variables.Variable'>


"""



