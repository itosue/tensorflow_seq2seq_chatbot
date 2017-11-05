import os
import tensorflow as tf
from sys import platform

tf.app.flags.DEFINE_boolean("use_small_data", False, "Use small data set if True")

FLAGS = tf.app.flags.FLAGS

is_fast_build = FLAGS.use_small_data

if is_fast_build:
    # this is for testing with real data
    MAX_ENC_VOCABULARY = 500# this should be >= 20
    NUM_LAYERS = 3
    LAYER_SIZE = 1024
    BATCH_SIZE = 16
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    # this is too minimum, but useful for quick debug
#    MAX_ENC_VOCABULARY = 20 # this should be >= 20
#    NUM_LAYERS = 2
#    LAYER_SIZE = 3
#    BATCH_SIZE = 4
#    buckets = [(5, 10)]
    beam_search = False
    beam_size = 2
else:
    MAX_ENC_VOCABULARY = 50000
    NUM_LAYERS = 3
    LAYER_SIZE = 1024
    BATCH_SIZE = 128
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    beam_search = True
    beam_size = 20



if platform == 'linux':
    GENERATED_DIR = os.getenv("HOME") + "/chatbot_generated/vocab_{}_layer_{}".format(MAX_ENC_VOCABULARY, LAYER_SIZE)
    LOGS_DIR = os.getenv("HOME") + "/chatbot_train_logs"
else:
    GENERATED_DIR = os.getenv("HOME") + "/Dropbox/tensorflow_seq2seq_chatbot/chatbot_generated/vocab_{}_layer_{}".format(MAX_ENC_VOCABULARY, LAYER_SIZE)
    LOGS_DIR = os.getenv("HOME") + "/chatbot_train_logs"


DATA_DIR = "data"
if is_fast_build:
    TWEETS_TXT = "{0}/tweets_short.txt".format(DATA_DIR)
else:
    TWEETS_TXT = "{0}/tweets3M.txt".format(DATA_DIR)


MAX_DEC_VOCABULARY = MAX_ENC_VOCABULARY

LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
MAX_GRADIENT_NORM = 5.0

TWEETS_TRAIN_ENC_IDX_TXT = "{0}/tweets_train_enc_idx.txt".format(GENERATED_DIR)
TWEETS_TRAIN_DEC_IDX_TXT = "{0}/tweets_train_dec_idx.txt".format(GENERATED_DIR)
TWEETS_VAL_ENC_IDX_TXT = "{0}/tweets_val_enc_idx.txt".format(GENERATED_DIR)
TWEETS_VAL_DEC_IDX_TXT = "{0}/tweets_val_dec_idx.txt".format(GENERATED_DIR)

VOCAB_ENC_TXT = "{0}/vocab_enc.txt".format(GENERATED_DIR)
VOCAB_DEC_TXT = "{0}/vocab_dec.txt".format(GENERATED_DIR)
