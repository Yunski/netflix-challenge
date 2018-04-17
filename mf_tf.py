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
        self.graph = tf.Graph()
        self.init_all_variables()
        self.inference()
        self._loss()


    def init_all_variables(self):
        with self.graph.as_default():
            self.mu = tf.constant(self.global_mean, name="global_mean", dtype=tf.float32)
            self.b_u = tf.Variable(tf.truncated_normal([self.n_users, 1], mean=0, stddev=1.0/self.n_features), name="user_bias")
            self.b_i = tf.Variable(tf.truncated_normal([self.n_items, 1], mean=0, stddev=1.0/self.n_features), name="item_bias")
            self.P = tf.Variable(tf.truncated_normal([self.n_users, self.n_features], mean=0, stddev=1.0/self.n_features), name="users")
            self.Q = tf.Variable(tf.truncated_normal([self.n_items, self.n_features], mean=0, stddev=1.0/self.n_features), name="items")
            self.acc_threshold = tf.constant(0.3, name="acc_threshold", dtype=tf.float32)
            self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
            self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
            self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")


    def fit(self, sess, data_indices, batch_size=None, max_iter=100, log_frequency=5, tol=1e-4, save_model=False):
        train_indices, val_indices, test_indices = data_indices
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
        train_op = optimizer.minimize(self.total_loss) 
        saver = tf.train.Saver()
        train_loss, prev_loss = 0.0, 0.0
        test_loss = 0.0
        sess.run(tf.global_variables_initializer())
        num_train_batches, num_val_batches, num_test_batches = 0, 0, 0
        if batch_size is not None:
            num_train_batches = len(train_indices[0]) / batch_size + int(len(train_indices[0]) % batch_size != 0)
            num_val_batches = len(val_indices[0]) / batch_size + int(len(val_indices[0]) % batch_size != 0)
            num_test_batches = len(test_indices[0]) / batch_size + int(len(test_indices[0]) % batch_size != 0) 
        for i in range(max_iter):
            iter_loss = 0.0
            if batch_size is not None:
                for j in range(num_train_batches):
                    start = j * batch_size
                    end = min(start+batch_size, len(train_indices[0]))
                    user_batch = train_indices[0][start:end]
                    item_batch = train_indices[1][start:end]
                    ratings_batch = train_indices[2][start:end]
                    _, loss = sess.run([train_op, self.train_loss], feed_dict={self.user_indices: user_batch, 
                                                                               self.item_indices: item_batch,
                                                                               self.ratings: ratings_batch})
                    iter_loss += loss
            else:
                _, iter_loss = sess.run([train_op, self.train_loss], feed_dict={self.user_indices: train_indices[0], 
                                                                                self.item_indices: train_indices[1],
                                                                                self.ratings: train_indices[2]})
            prev_loss = train_loss
            train_loss = np.sqrt(iter_loss / len(train_indices[0]))
            if (i+1) % log_frequency == 0:
                val_acc, val_loss = 0.0, 0.0
                if batch_size is not None:
                    for j in range(num_val_batches):
                        start = j * batch_size
                        end = min(start+batch_size, len(val_indices[0]))
                        user_batch = val_indices[0][start:end]
                        item_batch = val_indices[1][start:end]
                        ratings_batch = val_indices[2][start:end] 
                        val_batch_acc, val_batch_loss = sess.run([self.eval_acc, self.eval_loss], 
                                                                  feed_dict={self.user_indices: user_batch,
                                                                             self.item_indices: item_batch,
                                                                             self.ratings: ratings_batch})
                        val_acc += val_batch_acc*(end-start)
                        val_loss += val_batch_loss
                else:
                    val_acc, val_loss = sess.run([self.eval_acc, self.eval_loss], 
                                                  feed_dict={self.user_indices: val_indices[0],
                                                             self.item_indices: val_indices[1],
                                                             self.ratings: val_indices[2]})
                if batch_size is not None:
                    val_acc = val_acc / len(val_indices[0])
                val_loss = np.sqrt(val_loss / len(val_indices[0]))
                print("iter {} - train_loss: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}".format(i+1, train_loss, val_loss, val_acc))
            if np.abs(prev_loss - train_loss) <= tol:
                print("loss improvement <= tol")
                break
        print("Finished.")
        test_acc, test_loss = 0.0, 0.0
        if batch_size is not None:
            for j in range(num_test_batches):
                start = j * batch_size
                end = min(start+batch_size, len(test_indices[0]))
                user_batch = test_indices[0][start:end]
                item_batch = test_indices[1][start:end]
                ratings_batch = test_indices[2][start:end] 
                test_batch_acc, test_batch_loss = sess.run([self.eval_acc, self.eval_loss], 
                                                            feed_dict={self.user_indices: user_batch,
                                                                       self.item_indices: item_batch,
                                                                       self.ratings: ratings_batch})
                test_acc += test_batch_acc*(end-start)
                test_loss += test_batch_loss
        else: 
            test_acc, test_loss = sess.run([self.eval_acc, self.eval_loss], 
                                            feed_dict={self.user_indices: test_indices[0],
                                                       self.item_indices: test_indices[1],
                                                       self.ratings: test_indices[2]})
        if batch_size is not None: 
            test_acc = test_acc / len(test_indices[0])
        test_loss = np.sqrt(test_loss / len(test_indices[0]))
        print("test_loss: {:.4f} - test_acc: {:.4f}".format(test_loss, test_acc))
        
        if save_model:
            logs_path = saver.save(sess, "logs/mf/tf/model.ckpt")
            print("Model saved at {}".format(logs_path))                   
        return train_loss, test_loss


    def predict(self):
        P = tf.squeeze(tf.nn.embedding_lookup(self.P, self.user_indices))
        Q = tf.squeeze(tf.nn.embedding_lookup(self.Q, self.item_indices))
        prediction = tf.reduce_sum(tf.multiply(P, Q), axis=1) 
        self.prediction = tf.multiply(P, Q)
        b_u = tf.squeeze(tf.nn.embedding_lookup(self.b_u, self.user_indices))
        b_i = tf.squeeze(tf.nn.embedding_lookup(self.b_i, self.item_indices))
        prediction = self.mu + b_u + b_i + tf.squeeze(prediction)
        return prediction


    def inference(self):
        prediction = self.predict()
        acc = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(self.ratings - prediction), self.acc_threshold), tf.float32))
        loss = tf.reduce_sum(tf.square(self.ratings - prediction))
        self.eval_acc = acc
        self.eval_loss = loss
          

    def _reg(self):
        reg_loss = tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.b_u) + tf.nn.l2_loss(self.b_i)
        return self.reg * reg_loss

    
    def _loss(self):
        prediction = self.predict()
        self.train_loss = tf.reduce_sum(tf.square(self.ratings - prediction)) 
        loss = tf.nn.l2_loss(prediction - self.ratings) 
        reg_loss = self._reg()
        self.total_loss = loss + reg_loss
   