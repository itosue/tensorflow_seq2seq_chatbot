import math
import os
import sys

import numpy as np
import tensorflow as tf

import config
import data_processer
import lib.seq2seq_model as seq2seq_model


def show_progress(text):
    sys.stdout.write(text)
    sys.stdout.flush()


# P(reply|tweet)
def log_prob(session, model, enc_vocab, dec_vocab, tweet, reply):
    tweet_token_ids = data_processer.sentence_to_token_ids(tweet, enc_vocab)
    reply_token_ids = data_processer.sentence_to_token_ids(reply, dec_vocab)
    return model.log_prob(session, model, tweet_token_ids, reply_token_ids)


def log_prob_batch(session, model, enc_vocab, dec_vocab, tweets, replies):
    tweet_tokens_ids = [data_processer.sentence_to_token_ids(tweet, enc_vocab) for tweet in tweets]
    reply_tokens_ids = [data_processer.sentence_to_token_ids(reply, dec_vocab) for reply in replies]
    return model.log_prob_batch(session, model, tweet_tokens_ids, reply_tokens_ids)


def read_data_into_buckets(enc_path, dec_path, buckets):
    """Read tweets and reply and put them into buckets based on their length

    Args:
      enc_path: path to indexed tweets
      dec_path: path to indexed replies
      buckets: list of bucket

    Returns:
      data_set: data_set[i] has [tweet, reply] pairs for bucket[i]
    """
    # data_set[i] corresponds data for buckets[i]
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(enc_path, mode="r") as ef, tf.gfile.GFile(dec_path, mode="r") as df:
        tweet, reply = ef.readline(), df.readline()
        counter = 0
        while tweet and reply:
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in tweet.split()]
            target_ids = [int(x) for x in reply.split()]
            target_ids.append(data_processer.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                # Find bucket to put this conversation based on tweet and reply length
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            tweet, reply = ef.readline(), df.readline()
    for bucket_id in range(len(buckets)):
        print("  bucket{} size={}".format(buckets[bucket_id], len(data_set[bucket_id])))
    return data_set


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
def create_or_restore_model(session, buckets, forward_only, beam_search, beam_size, data_config):
    # beam search is off for training
    """Create model and initialize or load parameters"""
    print("Creating model...", flush=True)
    num_samples = 1024
    if config.is_fast_build:
        num_samples = config.MAX_ENC_VOCABULARY - 1

    with tf.variable_scope("swapped_model" if data_config.use_swapped_data else "normal_model"):
        model = seq2seq_model.Seq2SeqModel(source_vocab_size=config.MAX_ENC_VOCABULARY,
                                           target_vocab_size=config.MAX_DEC_VOCABULARY,
                                           buckets=buckets,
                                           num_samples=num_samples,
                                           size=config.LAYER_SIZE,
                                           num_layers=config.NUM_LAYERS,
                                           max_gradient_norm=config.MAX_GRADIENT_NORM,
                                           batch_size=config.BATCH_SIZE,
                                           learning_rate=config.LEARNING_RATE,
                                           learning_rate_decay_factor=config.LEARNING_RATE_DECAY_FACTOR,
                                           beam_search=beam_search,
                                           attention=True,
                                           forward_only=forward_only,
                                           beam_size=beam_size)

        ckpt = tf.train.get_checkpoint_state(data_config.generated_dir())
        # the checkpoint filename has changed in recent versions of tensorflow
        checkpoint_suffix = ".index"
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print("Loaded model parameters from %s" % ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    return model


def next_random_bucket_id(buckets_scale):
    n = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > n])
    return bucket_id


def train():
    # Only allocate 2/3 of the gpu memory to allow for running gpu-based predictions while training:
    #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    #    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    #    tf_config.gpu_options.allocator_type = 'BFC'

    # if we want to train based on (reply, tweet) pair instead of (tweet, reply pair).
    use_swapped_data = tf.app.flags.FLAGS.use_swapped_data
    data_config = config.DataConfig(use_swapped_data=use_swapped_data)

    should_train_with_rewards = tf.app.flags.FLAGS.use_rewards

    swapped_model = None
    if should_train_with_rewards:
        swapped_data_config = config.DataConfig(use_swapped_data=True)
        swapped_model_graph = tf.Graph()
        with swapped_model_graph.as_default():
            swapped_model_session = tf.Session(graph=swapped_model_graph)
            swapped_model = create_or_restore_model(swapped_model_session,
                                                    config.buckets,
                                                    forward_only=False,
                                                    beam_search=False,
                                                    beam_size=config.beam_size,
                                                    data_config=swapped_data_config)

    # with tf.Session(config=tf_config) as sess:
    with tf.Session() as sess:

        if use_swapped_data:
            show_progress("Using swapped data for training.")
        show_progress("Setting up data set for each buckets...\n")
        train_set = read_data_into_buckets(data_config.tweets_train_enc_idx_txt(),
                                           data_config.tweets_train_dec_idx_txt(),
                                           config.buckets)
        valid_set = read_data_into_buckets(data_config.tweets_val_enc_idx_txt(),
                                           data_config.tweets_val_dec_idx_txt(),
                                           config.buckets)
        show_progress("done\n")

        # False for train
        beam_search = False
        model = create_or_restore_model(sess, config.buckets, forward_only=False, beam_search=beam_search,
                                        beam_size=config.beam_size, data_config=data_config)

        # list of # of data in ith bucket
        train_bucket_sizes = [len(train_set[b]) for b in range(len(config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
        # This is for choosing randomly bucket based on distribution
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # Train Loop
        steps = 0
        previous_perplexities = []
        writer = tf.summary.FileWriter(data_config.logs_dir(), sess.graph)

        while True:
            bucket_id = next_random_bucket_id(train_buckets_scale)
            #            print(bucket_id)

            # Get batch
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            #      show_progress("Training bucket_id={0}...".format(bucket_id))

            # Train!
            #            _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
            #                                                           bucket_id,
            #                                                           forward_only=False,
            #                                                           beam_search=beam_search)
            _, average_perplexity, summary, _ = model.step_with_rewards(sess, swapped_model=swapped_model,
                                                                        encoder_inputs=encoder_inputs,
                                                                        decoder_inputs=decoder_inputs,
                                                                        target_weights=target_weights,
                                                                        bucket_id=bucket_id,
                                                                        forward_only=False,
                                                                        beam_search=beam_search)

            #      show_progress("done {0}\n".format(average_perplexity))

            steps = steps + 1
            if steps % 2 == 0:
                writer.add_summary(summary, steps)
                show_progress(".")
            if steps % 50 != 0:
                continue

            # check point
            checkpoint_path = os.path.join(data_config.generated_dir(), "seq2seq.ckpt")
            show_progress("\nSaving checkpoint...\n")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            show_progress("done\n")

            perplexity = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
            print("global step %d learning rate %.4f perplexity "
                  "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_perplexities) > 2 and perplexity > max(previous_perplexities[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_perplexities.append(perplexity)

            for bucket_id in range(len(config.buckets)):
                if len(valid_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % bucket_id)
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(valid_set, bucket_id)
                #                _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, beam_search=beam_search)
                _, average_perplexity, valid_summary, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                                     target_weights, bucket_id, True,
                                                                     beam_search=beam_search)
                writer.add_summary(valid_summary, steps)
                eval_ppx = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


tf.app.flags.DEFINE_boolean("use_rewards", False, "Train using rewards.")

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
