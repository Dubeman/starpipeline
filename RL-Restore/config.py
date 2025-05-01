import tensorflow as tf
import numpy as np

class AgentConfig(object):
    # train / test
    is_train = True

    # LSTM (minimal sizes)
    h_size = 32
    lstm_in = 24

    # test model
    play_model = 'models/'
    is_save = True

    # train model (minimal memory usage)
    save_dir = 'models/save/'
    model_log_dir = 'logs/'
    memory_size = 50000  # Drastically reduced
    learn_start = 1000   # Reduced
    test_step = 1000
    save_step = 50000
    max_step = 2000000
    target_q_update_step = 10000
    batch_size = 8       # Minimal batch size
    train_frequency = 8
    discount = 0.99
    
    # learning rate
    learning_rate = 0.0001
    learning_rate_minimum = 0.000025
    learning_rate_decay = 0.5
    learning_rate_decay_step = 1000000
    
    # experience replay
    ep_start = 1.
    ep_end = 0.1
    ep_end_t = 1000000

    # debug
    # learn_start = 500
    # test_step = 500
    # save_step = 1000
    # target_q_update_step = 1000

class EnvironmentConfig(object):
    # params for environment
    screen_width = 63
    screen_height = 63
    screen_channel = 3
    dataset = 'moderate'
    test_batch = 256     # Drastically reduced
    stop_step = 3
    reward_func = 'step_psnr_reward'

    # Astronomical image degradation parameters
    # Smaller blur range for preserving star points
    ASTRO_BLUR_RANGE = np.arange(0, 2.1, 0.2)  
    
    # Noise range suitable for astronomical imaging
    # Covers both readout noise and shot noise scenarios
    ASTRO_NOISE_RANGE = np.arange(0, 51, 5)  
    
    # Disable JPEG compression for astronomical images
    # Set to [100] to effectively disable compression
    ASTRO_JPG_QUALITY = [100]  

    # data path
    train_dir = 'data/train/'
    val_dir = 'data/valid/'
    test_dir = 'data/test/'


class DQNConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(FLAGS):
    config = DQNConfig()

    # TF version
    tf_version = tf.__version__.split('.')
    if int(tf_version[0]) >= 1 and int(tf_version[1]) > 4:  # TF version > 1.4
        for k in FLAGS:
            v = FLAGS[k].value
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        for k, v in FLAGS.__dict__['__flags'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Set additional attributes
    setattr(config, 'use_gpu', FLAGS.use_gpu)
    setattr(config, 'is_train', FLAGS.is_train)
    setattr(config, 'is_save', FLAGS.is_save)
    setattr(config, 'dataset', FLAGS.dataset)
    setattr(config, 'play_model', FLAGS.play_model)
    setattr(config, 'save_dir', FLAGS.save_dir)
    setattr(config, 'model_log_dir', FLAGS.model_log_dir)

    return config
