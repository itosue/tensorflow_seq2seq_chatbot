import tensorflow as tf

import config
import data_processer
import train


def main(_):
    data_config = config.DataConfig(use_swapped_data=False)
    swapped_data_config = config.DataConfig(use_swapped_data=True)
    model_graph = tf.Graph()
    smodel_graph = tf.Graph()
#    with tf.Session() as session:
    beam_search = False

    with model_graph.as_default():
        sess = tf.Session(graph=model_graph)
    # これがうまくいかないのは swapped_model のときに graph が normal になっているから？
        model = train.create_or_restore_model(sess,
                                              config.buckets,
                                              forward_only=False,  # This should work for both True and False
                                              beam_search=beam_search,  # False for simple debug
                                              beam_size=config.beam_size,
                                              data_config=data_config)

    with smodel_graph.as_default():
        ssess = tf.Session(graph=smodel_graph)
        swapped_model = train.create_or_restore_model(ssess,
                                                      config.buckets,
                                                      forward_only=False,  # This should work for both True and False
                                                      beam_search=beam_search,  # False for simple debug
                                                      beam_size=config.beam_size,
                                                      data_config=swapped_data_config)

        enc_vocab, _ = data_processer.initialize_vocabulary(data_config.vocab_enc_txt())
        dec_vocab, _ = data_processer.initialize_vocabulary(data_config.vocab_dec_txt())
        with sess.as_default():
            log_prob = train.log_prob(sess, model, enc_vocab, dec_vocab, "hige", "ますです")
            log_prob_batch = train.log_prob_batch(sess, model, enc_vocab, dec_vocab, ["hige"], ["ますです"])
    print(log_prob)
    print(log_prob_batch)


if __name__ == '__main__':
    tf.app.run()
