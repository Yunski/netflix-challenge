import tensorflow as tf

from mf_tf import MF

def restore():
    n_users = 2649430
    n_items = 17771
    n_features = 15
    #P = tf.Variable(tf.truncated_normal([self.n_users, self.n_features], mean=0, stddev=1.0/self.n_features), name="users")
    P = tf.get_variable("users", shape=[n_users, n_features])
    Q = tf.get_variable("items", shape=[n_items, n_features])
    b_u = tf.get_variable("user_bias", shape=[n_users, 1])
    b_i = tf.get_variable("item_bias", shape=[n_items, 1])
    mu = tf.constant(3.60429305089, name="global_mean", dtype=tf.float32)
    saver = tf.train.Saver({"users":P, "items":Q, "user_bias":b_u, "item_bias":b_i})
    with tf.Session() as sess: 
        saver.restore(sess, "logs/mf/tf/model.ckpt")
        #ratings = [(i, sess.run(tf.reduce_sum(tf.multiply(P[6,:], Q[i,:])) + mu + tf.squeeze(b_u)[6] + tf.squeeze(b_i)[i])) for i in range(1, 17771)]
        #print(list(sorted(ratings, key=lambda x: x[1]))[:10])
        
        q = sess.run(Q)
        
        print(q[[175, 197, 283, 299, 17479, 17525, 17541, 17526, 17627, 17480], 0:2])

if __name__ == '__main__':
    restore()