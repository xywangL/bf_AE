
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

filename = 'text8_1024_4.hash.list'
k = 4
M = 4096
lines = 253854
AEinput = 2048
AEoutput = 1024

learning_rate = 0.01
batch_size = 256
epoch = 10
n_batch = int(lines/batch_size)
print(n_batch)

file = open(filename,'r')

trainData = np.zeros((k,lines,AEoutput))


# In[2]:


count = 0;
while True:
    oneline = file.readline()
    if not oneline: break
    if oneline == '' or oneline == '\n': break
        
    nums = oneline.strip('\n')
    nums = nums.split(',')
    nums = tuple([int(i) for i in nums])
    k_count=0
    for n in nums:
        trainData[k_count][count][n] = 1
        k_count += 1
    count = count + 1
file.close()


# In[6]:


weights = tf.Variable(tf.random_normal([AEinput, AEoutput],stddev=0.01))
bias = tf.Variable(tf.zeros([AEoutput]))

weights_decode = tf.transpose(weights)
bias_decode = tf.Variable(tf.zeros([AEinput]))


# In[13]:


hash_input_1 = tf.placeholder(tf.float32, [None, AEoutput])
hash_input_2 = tf.placeholder(tf.float32, [None, AEoutput])
hash_input_3 = tf.placeholder(tf.float32, [None, AEoutput])
hash_input_4 = tf.placeholder(tf.float32, [None, AEoutput])
hash_init = tf.placeholder(tf.float32, [None, AEoutput])

##

input_layer_1 = tf.concat([hash_init, hash_input_1],1)
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer_1, weights), bias))

tmp1_decode_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights_decode), bias_decode))
loss_1 = tf.reduce_mean(tf.pow(tmp1_decode_layer1 - input_layer_1, 2))
optimizer_1 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)

##

input_layer_2 = tf.concat([layer1, hash_input_2],1)
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer_2, weights), bias))

tmp2_decode_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights_decode), bias_decode))
tmp2_decode2_input, _ = tf.split(tmp2_decode_layer1, 2, 1)
tmp2_decode_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(tmp2_decode2_input, weights_decode), bias_decode))
loss_2 = tf.reduce_mean(tf.pow(tmp2_decode_layer2 - input_layer_1, 2))
optimizer_2 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_2)

##

input_layer_3 = tf.concat([layer2, hash_input_3],1)
layer3 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer_3, weights), bias))

tmp3_decode_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, weights_decode), bias_decode))
tmp3_decode2_input, _ = tf.split(tmp3_decode_layer1, 2, 1)
tmp3_decode_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(tmp3_decode2_input, weights_decode), bias_decode))
tmp3_decode3_input, _ = tf.split(tmp3_decode_layer2, 2, 1)
tmp3_decode_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(tmp3_decode3_input, weights_decode), bias_decode))
loss_3 = tf.reduce_mean(tf.pow(tmp3_decode_layer3 - input_layer_1, 2))
optimizer_3 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_3)

##layer4 = encoded, decode_layer4 = decoded

input_layer_4 = tf.concat([layer3, hash_input_4],1)
layer4 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer_4, weights), bias))

decode_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(layer4, weights_decode), bias_decode))

input_decode_2, _ = tf.split(decode_layer1, [AEoutput, AEoutput], 1)
decode_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(input_decode_2, weights_decode), bias_decode))

input_decode_3, _ = tf.split(decode_layer2, [AEoutput, AEoutput], 1)
decode_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(input_decode_3, weights_decode), bias_decode))

input_decode_4, _ = tf.split(decode_layer3, [AEoutput, AEoutput], 1)
decode_layer4 = tf.nn.sigmoid(tf.add(tf.matmul(input_decode_3, weights_decode), bias_decode))

loss_full = tf.reduce_mean(tf.pow(decode_layer4 - input_layer_1, 2))
optimizer_full = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_full)
init = tf.global_variables_initializer()



# In[17]:


batch_size = 300
epoch = 5
n_batch = int(lines/batch_size)
X_init = np.zeros((batch_size, AEoutput))

encoded = np.zeros((lines,AEoutput))
decoded = np.zeros((lines,AEinput))

with tf.Session() as sess:

    sess.run(init)
    k_count = 1
    for j in range(epoch):
        total_loss = 0;

        for i in range(n_batch):
                    
            input_batch_1 = trainData[0][batch_size*i:batch_size*(i+1)]
            input_batch_2 = trainData[1][batch_size*i:batch_size*(i+1)]
            input_batch_3 = trainData[2][batch_size*i:batch_size*(i+1)]
            input_batch_4 = trainData[3][batch_size*i:batch_size*(i+1)]
            input_init = np.zeros((batch_size,AEoutput))

            input_feed = {hash_input_1: input_batch_1, hash_input_2: input_batch_2, hash_input_3: input_batch_3, hash_input_4: input_batch_4, hash_init: input_init}    
            

            _, loss_batch = sess.run([optimizer_1, loss_1], feed_dict=input_feed)
            total_loss += loss_batch

            #print('Step %i: Minibatch Loss: %f' % (i, loss_batch))
        print('k=%i  ---  Epoch %i: Loss: %f' % (k_count, j, total_loss/n_batch))
  
    print('============================================')
    
    ###
    
    k_count = 2
    for j in range(epoch):
        total_loss = 0;

        for i in range(n_batch):
                    
            input_batch_1 = trainData[0][batch_size*i:batch_size*(i+1)]
            input_batch_2 = trainData[1][batch_size*i:batch_size*(i+1)]
            input_batch_3 = trainData[2][batch_size*i:batch_size*(i+1)]
            input_batch_4 = trainData[3][batch_size*i:batch_size*(i+1)]
            input_init = np.zeros((batch_size,AEoutput))

            input_feed = {hash_input_1: input_batch_1, hash_input_2: input_batch_2, hash_input_3: input_batch_3, hash_input_4: input_batch_4, hash_init: input_init}    
            

            _, loss_batch = sess.run([optimizer_2, loss_2], feed_dict=input_feed)
            total_loss += loss_batch

            #print('Step %i: Minibatch Loss: %f' % (i, loss_batch))
        print('k=%i  ---  Epoch %i: Loss: %f' % (k_count, j, total_loss/n_batch))
  
    print('============================================')
    
    ###
    
    k_count = 3
    for j in range(epoch):
        total_loss = 0;

        for i in range(n_batch):
                    
            input_batch_1 = trainData[0][batch_size*i:batch_size*(i+1)]
            input_batch_2 = trainData[1][batch_size*i:batch_size*(i+1)]
            input_batch_3 = trainData[2][batch_size*i:batch_size*(i+1)]
            input_batch_4 = trainData[3][batch_size*i:batch_size*(i+1)]
            input_init = np.zeros((batch_size,AEoutput))

            input_feed = {hash_input_1: input_batch_1, hash_input_2: input_batch_2, hash_input_3: input_batch_3, hash_input_4: input_batch_4, hash_init: input_init}    
            

            _, loss_batch = sess.run([optimizer_3, loss_3], feed_dict=input_feed)
            total_loss += loss_batch

            #print('Step %i: Minibatch Loss: %f' % (i, loss_batch))
        print('k=%i  ---  Epoch %i: Loss: %f' % (k_count, j, total_loss/n_batch))
  
    print('============================================')
    
    ###
    k_count = 4
    for j in range(epoch):
        total_loss = 0;

        for i in range(n_batch):
                    
            input_batch_1 = trainData[0][batch_size*i:batch_size*(i+1)]
            input_batch_2 = trainData[1][batch_size*i:batch_size*(i+1)]
            input_batch_3 = trainData[2][batch_size*i:batch_size*(i+1)]
            input_batch_4 = trainData[3][batch_size*i:batch_size*(i+1)]
            input_init = np.zeros((batch_size,AEoutput))

            input_feed = {hash_input_1: input_batch_1, hash_input_2: input_batch_2, hash_input_3: input_batch_3, hash_input_4: input_batch_4, hash_init: input_init}    
            

            _, loss_batch = sess.run([optimizer_full, loss_full], feed_dict=input_feed)
            total_loss += loss_batch

            #print('Step %i: Minibatch Loss: %f' % (i, loss_batch))
        print('k=%i  ---  Epoch %i: Loss: %f' % (k_count, j, total_loss/n_batch))
  
    print('============================================')
    
    for i in range(n_batch):
                    
            input_batch_1 = trainData[0][batch_size*i:batch_size*(i+1)]
            input_batch_2 = trainData[1][batch_size*i:batch_size*(i+1)]
            input_batch_3 = trainData[2][batch_size*i:batch_size*(i+1)]
            input_batch_4 = trainData[3][batch_size*i:batch_size*(i+1)]
            input_init = np.zeros((batch_size,AEoutput))

            input_feed = {hash_input_1: input_batch_1, hash_input_2: input_batch_2, hash_input_3: input_batch_3, hash_input_4: input_batch_4, hash_init: input_init}    
            encoded[batch_size*i:batch_size*(i+1)], decoded[batch_size*i:batch_size*(i+1)] = sess.run([layer4,decode_layer4], feed_dict=input_feed)
            
    


# In[30]:


someone = 20
for i in range(1024):
    if decoded[someone][i] > 0.005:
        print('decode i = '+ str(i))
    if trainData[0][someone][i] > 0.1:
        print('train i = '+ str(i))

    

