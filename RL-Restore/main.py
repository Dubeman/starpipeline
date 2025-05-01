import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Disable eager execution for compatibility
from dqn.agent import Agent
from dqn.environment import MyEnvironment
from config import get_config
import sys
import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'

# Parameters
flags = tf.compat.v1.app.flags
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_boolean('is_train', False, 'Whether to do training or testing')
# test
flags.DEFINE_boolean('is_save', True, 'Whether to save results')
flags.DEFINE_string('dataset', 'moderate', 'Select a dataset from mild/moderate/severe')
flags.DEFINE_string('play_model', 'models/', 'Path for testing model')
# training
flags.DEFINE_string('save_dir', 'models/save/', 'Path for saving models')
flags.DEFINE_string('model_log_dir', 'logs/', 'Path for model logs')
FLAGS = flags.FLAGS


def main(_):
    with tf.compat.v1.Session() as sess:
        config = get_config(FLAGS)
        env = MyEnvironment(config)
        agent = Agent(config, env, sess)

        if FLAGS.is_train:
            agent.train()
        else:
            if FLAGS.dataset == 'mine':
                agent.play_mine()
            else:
                agent.play()


if __name__ == '__main__':
    tf.compat.v1.app.run()
