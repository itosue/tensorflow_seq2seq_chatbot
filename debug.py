import sys

import tensorflow as tf

import config
import train


# todo
# add comment about this
# make it work
# commit
# redup
# command line flag


def show_progress(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# assume training files are ready


def main():
    # with tf.Session(config=tf_config) as sess:
    with tf.Session() as sess:
        show_progress("Setting up data set for each buckets...")
        train_set = train.read_data_into_buckets(config.TWEETS_TRAIN_ENC_IDX_TXT, config.TWEETS_TRAIN_DEC_IDX_TXT,
                                                 config.buckets)
        valid_set = train.read_data_into_buckets(config.TWEETS_VAL_ENC_IDX_TXT, config.TWEETS_VAL_DEC_IDX_TXT,
                                                 config.buckets)
        show_progress("done\n")

        show_progress("Creating model...")
        # False for train
        beam_search = False
        model = train.create_or_restore_model(sess, config.buckets, forward_only=False, beam_search=beam_search,
                                              beam_size=config.beam_size)

        show_progress("{} done\n", model)


if __name__ == '__main__':
    main()
