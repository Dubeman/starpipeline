import cv2
import numpy as np
import os
import random
import scipy as sci
import scipy.io as sio
import tensorflow as tf
from .base import BaseModel
from .ops import linear, conv2d, clipped_error
from .replay_memory import ReplayMemory
from .utils import data_reformat, img2patch
from tqdm import tqdm

def convert_to_tf1_variable(var):
    """Convert a TF2 variable to TF1 style variable."""
    return tf.compat.v1.Variable(
        initial_value=var.numpy(),
        trainable=var.trainable,
        name=var.name,
        dtype=var.dtype
    )

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.cnn_format = 'NHWC'
        self.action_size = self.env.action_size
        
        # Build the network first
        self.build_dqn()
        
        # Initialize variables
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
        if self.is_train:
            self.memory = ReplayMemory(self.config)


    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        # Initialize LSTM states with dynamic shapes using placeholders for prediction network
        self.state_in = [
            tf.compat.v1.placeholder(tf.float32, [None, self.h_size], name='h_state'),
            tf.compat.v1.placeholder(tf.float32, [None, self.h_size], name='c_state')
        ]

        # Initialize LSTM states for target network
        self.t_state_in = [
            tf.compat.v1.placeholder(tf.float32, [None, self.h_size], name='target_h_state'),
            tf.compat.v1.placeholder(tf.float32, [None, self.h_size], name='target_c_state')
        ]

        # training network
        activation_fn = tf.nn.relu
        with tf.compat.v1.variable_scope('prediction'):
            # input batch and length for recurrent training
            self.batch = tf.compat.v1.placeholder(tf.int32, shape=[])
            self.length = tf.compat.v1.placeholder(tf.int32)
            self.s_t = tf.compat.v1.placeholder('float32', 
                [None, self.screen_height, self.screen_width, self.screen_channel],
                name='s_t')
            
            # CNN layers
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [9, 9], [2, 2],
                                                             activation_fn, self.cnn_format, name='l1')
            self.l1_out = self.l1
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l2')
            self.l2_out = self.l2
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l3')
            self.l3_out = self.l3
            self.l4, self.w['l4_w'], self.w['l4_b'] = conv2d(self.l3_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l4')
            shape = self.l4.get_shape().as_list()
            len_flat = 1
            for k in range(1, len(shape)):
                len_flat *= shape[k]
            self.l6_flat = tf.reshape(self.l4, [-1, len_flat])

            # Add action as input
            self.action_in = tf.compat.v1.placeholder('float32', [None, self.action_size - 1], name='action_in')
            self.action_out = self.action_in
            self.l7, self.w['l7_w'], self.w['l7_b'] = linear(self.l6_flat, self.lstm_in,
                                                             activation_fn=activation_fn, name='l7')
            self.l7_action = tf.concat([self.l7, self.action_out], 1)

            # Create rnn_input
            self.rnn_input = tf.reshape(self.l7_action, [self.batch, self.length,
                                                         self.l7_action.get_shape().as_list()[-1]])

            # Now create and run LSTM
            with tf.compat.v1.variable_scope('lstm'):
                lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                    num_units=self.h_size,
                    name='lstm_cell'
                )
                
                lstm_outputs, lstm_states = tf.compat.v1.nn.dynamic_rnn(
                    lstm_cell,
                    self.rnn_input,
                    initial_state=tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                        self.state_in[0], self.state_in[1]
                    ),
                    dtype=tf.float32
                )
                
                self.rnn = tf.reshape(lstm_outputs, shape=[-1, self.h_size])
                self.h_state = lstm_states.h
                self.c_state = lstm_states.c
                self.rnn_state = [self.h_state, self.c_state]

            # Continue with Q-value prediction
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.rnn, self.action_size, name='q')
            self.q_action = tf.argmax(self.q, axis=1)

        if self.is_train:
            # target network
            with tf.compat.v1.variable_scope('target'):
                # input batch and length for recurrent training
                self.t_batch = tf.compat.v1.placeholder(tf.int32, shape=[])
                self.t_length = tf.compat.v1.placeholder(tf.int32, shape=[])

                self.target_s_t = tf.compat.v1.placeholder('float32', [None, self.screen_height,
                                                                 self.screen_width, self.screen_channel], name='target_s_t')
                self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = \
                    conv2d(self.target_s_t, 32, [9, 9], [2, 2], activation_fn, self.cnn_format, name='target_l1')
                self.target_l1_out = self.target_l1
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = \
                    conv2d(self.target_l1_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l2')
                self.target_l2_out = self.target_l2
                self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = \
                    conv2d(self.target_l2_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l3')
                self.target_l3_out = self.target_l3
                self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    conv2d(self.target_l3_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l4')
                shape = self.target_l4.get_shape().as_list()
                target_len_flat = 1
                for k in range(1, len(shape)):
                    target_len_flat *= shape[k]
                self.target_l6_flat = tf.reshape(self.target_l4, [-1, target_len_flat])

                # Add action as input
                self.target_action_in = tf.compat.v1.placeholder('float32', [None, self.action_size - 1],
                                                       name='target_action_in')
                self.target_action_out = self.target_action_in
                self.target_l7, self.t_w['l7_w'], self.t_w['l7_b'] = \
                    linear(self.target_l6_flat, self.lstm_in, activation_fn=activation_fn, name='target_l7')
                self.target_l7_action = tf.concat([self.target_l7, self.target_action_out], 1)

                # add LSTM with dynamic steps
                self.t_rnn_input = tf.reshape(self.target_l7_action, [self.t_batch, self.t_length,
                                                                      self.target_l7_action.get_shape().as_list()[-1]])
                
                # Run target LSTM
                t_outputs, t_states = tf.compat.v1.nn.dynamic_rnn(
                    lstm_cell,
                    self.t_rnn_input,
                    initial_state=tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                        self.t_state_in[0], self.t_state_in[1]
                    ),
                    dtype=tf.float32
                )
                
                self.t_rnn = tf.reshape(t_outputs, shape=[-1, self.h_size])
                self.t_h_state = t_states.h
                self.t_c_state = t_states.c
                self.t_rnn_state = [self.t_h_state, self.t_c_state]
                
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = linear(self.t_rnn, self.action_size, name='target_q')
                self.target_q_idx = tf.compat.v1.placeholder('int32', [None, None], 'outputs_idx')
                self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

            # update target network
            with tf.compat.v1.variable_scope('pred_to_target'):
                self.t_w_input = {}
                self.t_w_assign_op = {}

                for name in self.w.keys():
                    self.t_w_input[name] = tf.compat.v1.placeholder('float32', self.t_w[name].shape.as_list(), name=name)
                    self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

            # optimizer
            with tf.compat.v1.variable_scope('optimizer'):
                self.global_step = tf.compat.v1.Variable(0, trainable=False)
                self.target_q_t = tf.compat.v1.placeholder('float32', [None], name='target_q_t')
                self.action = tf.compat.v1.placeholder('int64', [None], name='action')
                action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
                q_acted = tf.reduce_sum(self.q * action_one_hot, axis=1, name='q_acted')
                self.delta = self.target_q_t - q_acted
                self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

                # Learning rate setup
                self.learning_rate_step = tf.compat.v1.placeholder('int64', None, name='learning_rate_step')
                self.learning_rate = tf.compat.v1.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=False
                )
                self.learning_rate = tf.maximum(self.learning_rate_minimum, self.learning_rate)

                # Create optimizer
                self.opt = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08
                )
                
                # Ensure all variables are TF1 style
                trainable_vars = []
                for var in self.w.values():
                    if hasattr(var, '_in_graph_mode'):
                        trainable_vars.append(var)
                    else:
                        trainable_vars.append(convert_to_tf1_variable(var))
                
                # Create gradient accumulators
                self.grad_accum = [
                    tf.compat.v1.Variable(
                        tf.zeros(tv.get_shape().as_list(), dtype=tv.dtype),
                        trainable=False,
                        name=f"grad_accum_{i}"
                    ) for i, tv in enumerate(trainable_vars)
                ]
                
                # Zero ops
                self.zero_ops = [
                    grad_accum.assign(tf.zeros_like(grad_accum))
                    for grad_accum in self.grad_accum
                ]
                
                # Compute gradients
                grads_and_vars = self.opt.compute_gradients(
                    self.loss,
                    var_list=trainable_vars
                )
                
                # Accumulate gradients
                self.accum_ops = []
                for i, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        self.accum_ops.append(
                            self.grad_accum[i].assign_add(grad)
                        )
                
                # Apply gradients
                self.train_step = self.opt.apply_gradients(
                    [(self.grad_accum[i], var) for i, var in enumerate(trainable_vars)],
                    global_step=self.global_step
                )

            with tf.compat.v1.variable_scope('summary'):
                scalar_summary_tags = ['training.learning_rate']
                reward_tags = ['test.reward' + str(x + 1) for x in range(self.stop_step)] + ['test.reward_sum']
                scalar_summary_tags += reward_tags + reward_tags

                self.summary_placeholders = {}
                self.summary_ops = {}

                for tag in scalar_summary_tags:
                    self.summary_placeholders[tag] = tf.compat.v1.placeholder('float32', None, name=tag.replace(' ', '_'))
                    self.summary_ops[tag] = tf.compat.v1.summary.scalar("%s" % tag, self.summary_placeholders[tag])

                self.writer = tf.compat.v1.summary.FileWriter(self.model_log_dir, self.sess.graph)


    def train(self):
        start_step, self.update_count = 0, 0
        ep_reward, total_reward, self.total_loss, self.total_q = 0., 0., 0., 0.
        max_avg_ep_reward = -1
        ep_rewards, actions = [], []
        img, reward, action, terminal = self.env.new_image()  # get the first input image
        self.state_explore = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))  # initialize LSTM states

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                self.update_count, ep_reward = 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. agent predicts an action
            action = self.predict(img)
            self.pre_action = action

            # 2. environment conducts the action
            img, reward, terminal = self.env.act(action)

            # 3. observe (add memory, train)
            self.observe(img, reward, action, terminal)

            if terminal:
                img, _, _, terminal = self.env.new_image()
                self.state_explore = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))  # initialize LSTM states
                ep_reward += reward
                ep_rewards.append(ep_reward)
                ep_reward = 0.

                # add the initial state
                self.memory.add(img, 0., -1, terminal)  # action -1=255 for uint8
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            # validation and logging
            if self.step >= self.learn_start and self.step % self.test_step == self.test_step - 1:
                avg_reward = total_reward / self.test_step
                avg_loss = self.total_loss / self.update_count
                avg_q = self.total_q / self.update_count
                avg_ep_reward = np.mean(ep_rewards)

                print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f' 
                      % (avg_reward, avg_loss, avg_q, avg_ep_reward))

                if self.step % self.save_step == self.save_step - 1 and max_avg_ep_reward * 0.9 <= avg_ep_reward:
                    self.save_model(self.step + 1)
                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                # validation
                reward_test_vec = []
                psnr_test_vec = []
                reward_str = ''
                psnr_str = ''
                reward_sum = 0.
                for cur_step in range(self.stop_step):
                    action_test = self.predict_test(count_step=cur_step)
                    self.pre_action_test = action_test.copy()
                    reward_test, psnr_test, base_psnr = self.env.act_test(action_test, step=cur_step)
                    reward_sum += reward_test
                    reward_test_vec.append(reward_test)
                    psnr_test_vec.append(psnr_test)
                    reward_str += 'reward' + str(cur_step + 1) + ': %.4f, '
                    psnr_str += 'psnr' + str(cur_step + 1) + ': %.4f, '
                reward_str += 'reward_sum: %.4f, '
                reward_test_vec.append(reward_sum)
                print_str = reward_str + psnr_str + 'base_psnr: %.4f'
                print(print_str % tuple(reward_test_vec + psnr_test_vec + [base_psnr]))

                # logging (write summary)
                diction = {'training.learning_rate': self.learning_rate.eval({self.learning_rate_step: self.step})}
                for cur_step in range(self.stop_step):
                    diction['test.reward' + str(cur_step + 1)] = reward_test_vec[cur_step]
                diction['test.reward_sum'] = reward_test_vec[-1]
                self.inject_summary(diction)

                total_reward = 0.
                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                ep_reward = 0.
                ep_rewards = []
                actions = []


    def predict(self, s_t, test_ep=None):
        action_size = self.action_size
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(action_size)
        else:
            count_step = self.env.count
            action_in = np.zeros([1, action_size - 1])
            if count_step > 0:
                action_in[0, self.pre_action] = 1.
            action_vec, self.state_explore = self.sess.run([self.q, self.rnn_state],
                                                          {self.s_t: s_t, 
                                                           self.action_in: action_in,
                                                           self.batch: 1, 
                                                           self.length: 1,
                                                           self.state_in[0]: self.state_explore[0],
                                                           self.state_in[1]: self.state_explore[1]})
            action_vec = action_vec[0]
            action = action_vec.argmax(axis=0)

        return action


    def observe(self, screen, reward, action, terminal):
        # add memory
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            # train agent network
            if self.step % self.train_frequency == 0:
                self.q_learning_lstm_batch()

            # update target network
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()


    def q_learning_lstm_batch(self):
        # assign accumulated gradients to zero
        self.sess.run(self.zero_ops)

        # initialization
        action_size = self.action_size
        temp_q_t = 0.
        temp_loss = 0.

        # get a batch of episodes (toolchains)
        s_t, action, reward = self.memory.getEpiBatch(self.batch_size)

        # loop for different lengths of episode
        for m in range(self.stop_step):
            s_t_cur = s_t[m]
            action_cur = action[m]
            reward_cur = reward[m]
            num = len(reward_cur)
            if num < 1:
                continue
            rnn_length = m + 1
            rnn_batch = num // rnn_length

            # derive action at previous step
            action_in = np.zeros([num, action_size - 1])
            for k in range(rnn_batch):
                for p in range(rnn_length):
                    if p > 0:
                        idx = k * rnn_length + p
                        action_in[idx, action_cur[idx - 1]] = 1.

            # Initialize LSTM states with zeros
            initial_h_state = np.zeros([rnn_batch, self.h_size])
            initial_c_state = np.zeros([rnn_batch, self.h_size])

            # target Q value (denoted as y_t in Eq. (6))
            q_t_plus_1 = self.target_q.eval({
                self.target_s_t: s_t_cur,
                self.target_action_in: action_in,
                self.t_batch: rnn_batch,
                self.t_length: rnn_length,
                self.t_state_in[0]: initial_h_state,
                self.t_state_in[1]: initial_c_state
            })
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            for k in range(rnn_batch):
                idx = k * rnn_length
                max_q_t_plus_1[idx: idx + rnn_length - 1] = max_q_t_plus_1[idx + 1: idx + rnn_length]
                max_q_t_plus_1[idx + rnn_length - 1] = 0.
            target_q_t = self.discount * max_q_t_plus_1 + reward_cur

            # Compute gradients using gradient tape
            with tf.GradientTape() as tape:
                loss_value = self.loss
            grads = tape.gradient(loss_value, list(self.w.values()))
            gvs = list(zip(grads, list(self.w.values())))
            
            # accumulate gradients
            grad_weight = 1.0 * rnn_batch / self.batch_size  # gradient weight
            q_t, delta, loss = self.sess.run([self.q, self.delta, self.loss],
                                      {
                                          self.target_q_t: target_q_t,
                                          self.action: action_cur,
                                          self.s_t: s_t_cur,
                                          self.learning_rate_step: self.step,
                                          self.action_in: action_in,
                                          self.batch: rnn_batch,
                                          self.length: rnn_length,
                                          self.state_in[0]: initial_h_state,  # Add prediction network states
                                          self.state_in[1]: initial_c_state
                                      })
            
            # Apply gradients with accumulation
            for i, (grad, var) in enumerate(gvs):
                if grad is not None:
                    self.sess.run(self.grad_accum[i].assign_add(grad * grad_weight))
            
            temp_q_t += q_t.mean()
            temp_loss += loss

        # apply accumulated gradients
        self.sess.run(self.train_step, {self.learning_rate_step: self.step})

        # update statistics
        self.total_loss += temp_loss
        self.total_q += q_t.mean() / self.batch_size
        self.update_count += 1


    def update_target_q_network(self):
        for name in self.w.keys():
            value = self.sess.run(self.w[name])  # Get value using session
            self.sess.run(self.t_w_assign_op[name], 
                         feed_dict={self.t_w_input[name]: value})  # Update using session


    def inject_summary(self, tag_dict):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()],
                                          {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step + 1)


    def predict_test(self, count_step = 0):
        if count_step == 0:
            imgs = self.env.get_data_test()
            env_steps = np.zeros(len(imgs), dtype=int)
            # initialize LSTM states
            self.state_test = (np.zeros([len(imgs), self.h_size]), np.zeros([len(imgs), self.h_size]))
            self.sess_test = self.sess  # Use the session we created in __init__
        else:
            imgs = self.env.get_test_imgs()
            env_steps = self.env.get_test_steps()

        action_in = np.zeros([len(imgs), self.action_size - 1])
        if count_step > 0:
            for k in range(len(imgs)):
                if self.pre_action_test[k] < self.action_size - 1:
                    action_in[k, self.pre_action_test[k]] = 1.

        actions_vec, self.state_test = self.sess_test.run([self.q, self.rnn_state],
                                       {self.s_t: imgs, 
                                        self.action_in: action_in, 
                                        self.state_in[0]: self.state_test[0],  # Split state input
                                        self.state_in[1]: self.state_test[1],
                                        self.batch: len(imgs), 
                                        self.length: 1})
        actions = actions_vec.argmax(axis=1)

        # choose the last action if pre-action is the last
        if count_step > 0:
            actions[self.pre_action_test==self.action_size - 1] = self.action_size - 1
        actions[env_steps == self.stop_step] = self.action_size - 1  # already stopped before

        return actions


    def predict_mine(self, my_img, pre_action=None, count_step=0):
        # image --> patches
        my_patches = img2patch(my_img)
        num_patch = len(my_patches)

        if count_step == 0:
            # initialize LSTM states
            self.state_test = (np.zeros([num_patch, self.h_size]), np.zeros([num_patch, self.h_size]))
            self.sess_test = tf.get_default_session()

        # set action at the previous step as input
        action_in = np.zeros([num_patch, self.action_size - 1])
        if count_step > 0:
            assert pre_action is not None and pre_action != self.action_size - 1
            action_in[:, pre_action] = 1.

        # run the agent
        actions_vec, self.state_test = self.sess_test.run([self.q, self.rnn_state],
                                                          {self.s_t: my_patches, self.action_in: action_in,
                                                           self.state_in: self.state_test,
                                                           self.batch: num_patch, self.length: 1})
        actions = actions_vec.argmax(axis=1)

        # vote
        action = np.bincount(actions).argmax()

        return action


    def play(self):  # test
        rewards = []
        actions = []
        psnrs = []
        diction = {}
        # create folder if needed
        if self.is_save:
            names = []
            save_path = 'results/' + self.dataset + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # loop for test batch
        test_update = True
        total_base_psnr = 0.
        name_list = self.env.name_list
        while test_update:
            img_num, batch_size = self.env.get_test_info()  # current image No. and batch size
            for m in range(self.stop_step):
                # predict action
                action_test = self.predict_test(count_step=m)
                self.pre_action_test = action_test.copy()
                reward_all, psnr_all, base_psnr = self.env.act_test(action_test, step=m)
                if m == 0:
                    total_base_psnr += base_psnr * batch_size
                    # initialize names
                    if self.is_save:
                        for k in range(batch_size):
                            names.append(save_path + name_list[k + img_num])

                # store reward, psnr, action
                rewards.append(reward_all)
                actions.append(action_test)
                psnrs.append(psnr_all)

                # construct dictionary for log
                diction['reward' + str(m + 1)] = rewards[-1] if 'reward' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['reward' + str(m + 1)], rewards[-1]], axis=0)
                diction['action' + str(m + 1)] = actions[-1] if 'action' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['action' + str(m + 1)], actions[-1]], axis=0)
                diction['psnr' + str(m + 1)] = psnrs[-1] if 'psnr' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['psnr' + str(m + 1)], psnrs[-1]], axis=0)

                # print results
                print(('reward' + str(m + 1) + ': %.4f, psnr' + str(m + 1) + ': %.4f' +
                       ', tested images: %d, total tested images: %d') % (reward_all.mean(), psnr_all.mean(),
                                                                          batch_size, img_num + batch_size))

                # save images
                if self.is_save:
                    cur_img = self.env.get_test_imgs()
                    save_img = data_reformat(cur_img)
                    for k in range(batch_size):
                        names[k + img_num] += '_' + str(action_test[k] + 1)
                        cv2.imwrite(names[k + img_num] + '.png', 255 * save_img[k, ...])

            test_update = self.env.update_test_data()

        # print final results
        print('This is the final result:')
        for m in range(self.stop_step):
            print(('reward' + str(m + 1) + ': %.4f, psnr' + str(m + 1) + ': %.4f' +
                   ', total tested images: %d') % (diction['reward' + str(m + 1)].mean(),
                                                  diction['psnr' + str(m + 1)].mean(),
                                                  img_num + batch_size))
        mean_base_psnr = total_base_psnr / (img_num + batch_size)
        print('base_psnr: %.4f' % mean_base_psnr)


    def play_mine(self):
        # create save folder if needed
        if self.is_save:
            save_path = 'results/' + self.dataset + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        my_img, base_name = self.env.update_test_mine()
        while my_img is not None:
            for m in range(self.stop_step):
                # first step
                if m == 0:
                    # previous action is None
                    pre_action = None
                    # initialize names
                    if self.is_save:
                        name = save_path + base_name
                        # save the initial image
                        cv2.imwrite(name + '.png', my_img[:,:,::-1] * 255)


                # predict action
                action = self.predict_mine(my_img, pre_action, count_step=m)
                if action == self.action_size - 1:
                    break
                pre_action = action
                my_img_next = self.env.act_test_mine(my_img, action)
                my_img = my_img_next

                # save images
                if self.is_save:
                    name += '_' + str(action + 1)
                    save_img = my_img[:,:,::-1] * 255
                    cv2.imwrite(name + '.png', save_img)

            # update my test image
            print('Image %s processed' % base_name)
            my_img, base_name = self.env.update_test_mine()

        print('Done!')
        return


