import sys

import tensorflow as tf

import config
import train


# todo
# add comment about this
# flag for processing small data set
# make it work
# commit
# redup
# command line flag


def main(argv):
    with tf.Session() as sess:
        # False for train
        beam_search = False
        model = train.create_or_restore_model(sess, config.buckets, forward_only=False, beam_search=beam_search,
                                              beam_size=config.beam_size)
        print(model)

if __name__ == '__main__':
    tf.app.run()
