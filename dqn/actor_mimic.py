import tensorflow as tf

from agent import Agent
from .history import History
from .ops import clipped_error, conv2d, linear
from .replay_memory import ReplayMemory



class AMN(Agent):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'

        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def build_dqn(self):

        self.w = {}
        self.t_w = {}

        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                self.s_t = tf.placeholder('float32',
                                          [None, self.screen_height, self.screen_width, self.history_length],
                                          name='s_t')
            else:
                self.s_t = tf.placeholder('float32',
                                          [None, self.history_length, self.screen_height, self.screen_width],
                                          name='s_t')

            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                                                             32, [8, 8], [4, 4], initializer, activation_fn,
                                                             self.cnn_format, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn,
                                                             self.cnn_format, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             self.cnn_format, name='l3')

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] =\
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] =\
                    linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

                self.value, self.w['val_w_out'], self.w['val_w_b'] =\
                    linear(self.value_hid, 1, name='value_out')

                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] =\
                    linear(self.adv_hid, self.env.action_size, name='adv_out')

                # Average Dueling
                self.q = self.value + (self.advantage -
                                       tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn,
                                                                 name='l4')
                self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in xrange(self.env.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target'):
            if self.cnn_format == 'NHWC':
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.screen_height, self.screen_width, self.history_length],
                                                 name='target_s_t')
            else:
                self.target_s_t = tf.placeholder('float32',
                                                 [None, self.history_length, self.screen_height, self.screen_width],
                                                 name='target_s_t')

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
                                                                        32, [8, 8], [4, 4], initializer, activation_fn,
                                                                        self.cnn_format, name='target_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
                                                                        64, [4, 4], [2, 2], initializer, activation_fn,
                                                                        self.cnn_format, name='target_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
                                                                        64, [3, 3], [1, 1], initializer, activation_fn,
                                                                        self.cnn_format, name='target_l3')

            shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            if self.dueling:
                self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] =\
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

                self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] =\
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] =\
                    linear(self.t_value_hid, 1, name='target_value_out')

                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] =\
                    linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage -
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] =\
                    linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] =\
                    linear(self.target_l4, self.env.action_size, name='target_q')

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')



            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                       self.learning_rate,
                                                       self.learning_rate_step,
                                                       self.learning_rate_decay_step,
                                                       self.learning_rate_decay,
                                                       staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                    self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',\
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward',
                                   'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag),
                                                          self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

        tf.initialize_all_variables().run()

        self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

        self.load_model()
        self.update_target_q_network()
