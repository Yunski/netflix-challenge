import numpy as np
import tensorflow as tf

"""
References https://github.com/mesuvash/TFMF/
"""

class MF(object):
    def __init__(self, n_users, n_items, n_features, global_mean, learning_rate=0.01, reg=0.1):
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.global_mean = global_mean
        self.learning_rate = learning_rate
        self.reg = reg
        self.init_all_variables()
        

    def init_all_variables(self):
        self.mu = tf.constant(self.global_mean, name="global_mean", dtype=tf.float32)
        self.b_u = tf.Variable(tf.truncated_normal([self.n_users, 1], mean=0, stddev=1.0/self.n_features), name="user_bias")
        self.b_i = tf.Variable(tf.truncated_normal([self.n_items, 1], mean=0, stddev=1.0/self.n_features), name="item_bias")
        self.P = tf.Variable(tf.truncated_normal([self.n_users, self.n_features], mean=0, stddev=1.0/self.n_features), name="users")
        self.Q = tf.Variable(tf.truncated_normal([self.n_items, self.n_features], mean=0, stddev=1.0/self.n_features), name="items")
             

    def fit(self, train_indices, test_indices, max_iter=1000, log_frequency=10, tol=1e-4, save_model=False):
        cost = self.loss(train_indices)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
        train_op = optimizer.minimize(cost) 
        saver = tf.train.Saver()
        cur_loss, prev_loss = 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(max_iter):
                _, loss = sess.run([train_op, self.train_loss])
                prev_loss = cur_loss
                cur_loss = loss

                if (i+1) % log_frequency == 0:
                    test_acc, test_loss = self.eval(test_indices)
                    test_acc, test_loss = sess.run(test_acc), sess.run(test_loss)
                    print("iter {} - train_loss: {:.4f} - test_loss: {:.4f} - test_acc: {:.4f}".format(i, loss, test_loss, test_acc))
                if np.abs(prev_loss - cur_loss) <= tol:
                    print(prev_loss, cur_loss)
                    print("loss improvement <= tol")
                    break
            if save_model:
                logs_path = saver.save(sess, "logs/mf/tf/model.ckpt")
                print("Model saved at {}".format(logs_path))                   
        return cur_loss


    def predict(self, user_indices, item_indices):
        P = tf.squeeze(tf.nn.embedding_lookup(self.P, user_indices))
        Q = tf.squeeze(tf.nn.embedding_lookup(self.Q, item_indices))
        prediction = tf.reduce_sum(tf.multiply(P, Q), axis=1) 
        b_u = tf.squeeze(tf.nn.embedding_lookup(self.b_u, user_indices))
        b_i = tf.squeeze(tf.nn.embedding_lookup(self.b_i, item_indices))
        prediction = self.mu + b_u + b_i + tf.squeeze(prediction)
        return prediction


    def eval(self, test_indices, acc_threshold=0.3):
        user_indices, item_indices, ratings = test_indices
        prediction = self.predict(user_indices, item_indices)
        acc = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(ratings - prediction), tf.constant(0.3)), tf.float32))
        loss = tf.sqrt(tf.nn.l2_loss(ratings - prediction) * 2.0 / len(ratings)) 
        return acc, loss
          

    def _reg(self):
        reg_loss = tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.b_u) + tf.nn.l2_loss(self.b_i)
        return self.reg * reg_loss

    
    def loss(self, indices):
        user_indices, item_indices, ratings = indices
        prediction = self.predict(user_indices, item_indices)
        res = tf.nn.l2_loss(prediction - ratings) 
        self.train_loss = tf.sqrt(res*2.0 / len(ratings))
        reg_loss = self._reg()
        self.total_loss = res + reg_loss
        return self.total_loss
   