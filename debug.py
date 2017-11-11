import tensorflow as tf

import config
import data_processer
import train


def main(_):
    data_config = config.DataConfig(use_swapped_data=False)
    swapped_data_config = config.DataConfig(use_swapped_data=True)
    with tf.Session() as session:
        beam_search = False

        with tf.variable_scope("foo"):
            model = train.create_or_restore_model(session,
                                                  config.buckets,
                                                  forward_only=False,  # This should work for both True and False
                                                  beam_search=beam_search,  # False for simple debug
                                                  beam_size=config.beam_size,
                                                  data_config=data_config)
        with tf.variable_scope("bar"):
            swapped_model = train.create_or_restore_model(session,
                                                          config.buckets,
                                                          forward_only=False,  # This should work for both True and False
                                                          beam_search=beam_search,  # False for simple debug
                                                          beam_size=config.beam_size,
                                                          data_config=swapped_data_config)



        enc_vocab, _ = data_processer.initialize_vocabulary(data_config.vocab_enc_txt())
        dec_vocab, _ = data_processer.initialize_vocabulary(data_config.vocab_dec_txt())
        log_prob = train.log_prob(session, model, enc_vocab, dec_vocab, "hige", "ますです")
        print(log_prob)


if __name__ == '__main__':
    tf.app.run()
