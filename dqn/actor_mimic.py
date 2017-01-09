import tensorflow as tf
import time
from tqdm import tqdm

from agent import Agent
from .history import History
from .ops import clipped_error, conv2d, linear
from .replay_memory import ReplayMemory



class AMN(Agent):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'
        self.weight_e1_dir = 'weights_breakout'
        self.weight_e2_dir = 'weights_pong'

        self.env1 = environment
        self.env2 = environment
        self.history1 = History(self.config)
        self.history2 = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def train(self):
        start_step = self.step_op.eval()
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen1, reward1, action1, terminal1 = self.env1.new_random_game()
        screen2, reward2, action2, terminal2 = self.env2.new_random_game()

        for _ in range(self.history_length):
            self.history1.add(screen1)
            self.history2.add(screen2)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())
            # 2. act
            # TODO : Change self.env to self.env[np.random_int(len(envs))
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                          % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': num_game,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                        }, self.step)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

    def build_dqn(self):

        self.w = {}
        self.t_w = {}

        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # expert1 network
        with tf.variable_scope('Expert1'):
            if self.cnn_format == 'NHWC':
                self.e1_s_t = tf.placeholder('float32',
                                          [None, self.screen_height, self.screen_width, self.history_length],
                                          name='e1_s_t')
            else:
                self.e1_s_t = tf.placeholder('float32',
                                          [None, self.history_length, self.screen_height, self.screen_width],
                                          name='e1_s_t')

            self.e1_l1, self.e1_w['e1_l1_w'], self.e1_w['e1_l1_b'] = conv2d(self.s_t,
                                                             32, [8, 8], [4, 4], initializer, activation_fn,
                                                             self.cnn_format, name='e1_l1')
            self.e1_l2, self.e1_w['l2_w'], self.e1_w['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn,
                                                             self.cnn_format, name='e1_l2')
            self.e1_l3, self.e1_w['e1_l3_w'], self.e1_w['e1_l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             self.cnn_format, name='e1_l3')
            shape = self.e1_l3.get_shape().as_list()
            self.e1_l3_flat = tf.reshape(self.e1_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.e1_l4, self.e1_w['e1_l4_w'], self.e1_w['e1_l4_b'] = linear(self.e1_l3_flat, 512, activation_fn=activation_fn,
                                                             name='e1_l4')
            self.e1_q, self.e1_w['e1_q_w'], self.e1_w['e1_q_b'] = linear(self.e1_l4, self.env.action_size, name='q')
            self.e1_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in xrange(self.env.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # expert2 network
        with tf.variable_scope('Expert1'):
            if self.cnn_format == 'NHWC':
                self.e2_s_t = tf.placeholder('float32',
                                          [None, self.screen_height, self.screen_width, self.history_length],
                                          name='e2_s_t')
            else:
                self.e2_s_t = tf.placeholder('float32',
                                          [None, self.history_length, self.screen_height, self.screen_width],
                                          name='e2_s_t')

            self.e2_l1, self.e2_w['e2_l1_w'], self.e2_w['e2_l1_b'] = conv2d(self.s_t,
                                                             32, [8, 8], [4, 4], initializer, activation_fn,
                                                             self.cnn_format, name='e2_l1')
            self.e2_l2, self.e2_w['l2_w'], self.e2_w['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn,
                                                             self.cnn_format, name='e2_l2')
            self.e2_l3, self.e2_w['e2_l3_w'], self.e2_w['e2_l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn,
                                                             self.cnn_format, name='e2_l3')
            shape = self.e2_l3.get_shape().as_list()
            self.e2_l3_flat = tf.reshape(self.e2_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            self.e2_l4, self.e2_w['e2_l4_w'], self.e2_w['e2_l4_b'] = linear(self.e2_l3_flat, 512, activation_fn=activation_fn,
                                                             name='e2_l4')
            self.e2_q, self.e2_w['e2_q_w'], self.e2_w['e2_q_b'] = linear(self.e2_l4, self.env.action_size, name='q')
            self.e2_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in xrange(self.env.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

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

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()