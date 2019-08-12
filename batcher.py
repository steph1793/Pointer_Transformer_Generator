import tensorflow as tf
import glob

from data_helper import Vocab, Data_Helper


def _parse_function(example_proto):
	# Create a description of the features.
	feature_description = {
	'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
	'abstract': tf.io.FixedLenFeature([], tf.string, default_value='')
	}
	# Parse the input `tf.Example` proto using the dictionary above.
	parsed_example = tf.io.parse_single_example(example_proto, feature_description)

	return parsed_example



def example_generator(filenames, vocab_path, vocab_size, max_enc_len, max_dec_len, training=False):

	raw_dataset = tf.data.TFRecordDataset(filenames)
	parsed_dataset = raw_dataset.map(_parse_function)
	if training:
		parsed_dataset = parsed_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()


	vocab = Vocab(vocab_path, vocab_size)

	for raw_record in parsed_dataset:

		article = raw_record["article"].numpy().decode()
		abstract = raw_record["abstract"].numpy().decode()

		start_decoding = vocab.word_to_id(vocab.START_DECODING)
		stop_decoding = vocab.word_to_id(vocab.STOP_DECODING)

		article_words = article.split()[ : max_enc_len]
		enc_len = len(article_words)
		enc_input = [vocab.word_to_id(w) for w in article_words]
		enc_input_extend_vocab, article_oovs = Data_Helper.article_to_ids(article_words, vocab)

		abstract_sentences = [sent.strip() for sent in Data_Helper.abstract_to_sents(abstract)]
		abstract = ' '.join(abstract_sentences)
		abstract_words = abstract.split()
		abs_ids = [vocab.word_to_id(w) for w in abstract_words]
		abs_ids_extend_vocab = Data_Helper.abstract_to_ids(abstract_words, vocab, article_oovs)
		dec_input, target = Data_Helper.get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
		_, target = Data_Helper.get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
		dec_len = len(dec_input)

		output = {
			"enc_len":enc_len,
			"enc_input" : enc_input,
			"enc_input_extend_vocab"  : enc_input_extend_vocab,
			"article_oovs" : article_oovs,
			"dec_input" : dec_input,
			"target" : target,
			"dec_len" : dec_len,
			"article" : article,
			"abstract" : abstract,
			"abstract_sents" : abstract_sentences
		}


		yield output


def batch_generator(generator, filenames, vocab_path, vocab_size, max_enc_len, max_dec_len, batch_size, training):

	dataset = tf.data.Dataset.from_generator(generator, args = [filenames, vocab_path, vocab_size, max_enc_len, max_dec_len, training],
											output_types = {
												"enc_len":tf.int32,
												"enc_input" : tf.int32,
												"enc_input_extend_vocab"  : tf.int32,
												"article_oovs" : tf.string,
												"dec_input" : tf.int32,
												"target" : tf.int32,
												"dec_len" : tf.int32,
												"article" : tf.string,
												"abstract" : tf.string,
												"abstract_sents" : tf.string
											}, output_shapes={
												"enc_len":[],
												"enc_input" : [None],
												"enc_input_extend_vocab"  : [None],
												"article_oovs" : [None],
												"dec_input" : [None],
												"target" : [None],
												"dec_len" : [],
												"article" : [],
												"abstract" : [],
												"abstract_sents" : [None]
											})
	dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len":[],
												"enc_input" : [None],
												"enc_input_extend_vocab"  : [None],
												"article_oovs" : [None],
												"dec_input" : [max_dec_len],
												"target" : [max_dec_len],
												"dec_len" : [],
												"article" : [],
												"abstract" : [],
												"abstract_sents" : [None]}),
											padding_values={"enc_len":-1,
												"enc_input" : 1,
												"enc_input_extend_vocab"  : 1,
												"article_oovs" : b'',
												"dec_input" : 1,
												"target" : 1,
												"dec_len" : -1,
												"article" : b"",
												"abstract" : b"",
												"abstract_sents" : b''},
											drop_remainder=True)
	def update(entry):
		return ({"enc_input" : entry["enc_input"],
			"extended_enc_input" : entry["enc_input_extend_vocab"],
			"article_oovs" : entry["article_oovs"],
			"enc_len" : entry["enc_len"],
			"article" : entry["article"],
			"max_oov_len" : tf.shape(entry["article_oovs"])[1] },

			{"dec_input" : entry["dec_input"],
			"dec_target" : entry["target"],
			"dec_len" : entry["dec_len"],
			"abstract" : entry["abstract"]})


	dataset = dataset.map(update)

	return dataset


def batcher(data_path, vocab_path, hpm):
  
	filenames = glob.glob("{}/*.tfrecords".format(data_path))
	dataset = batch_generator(example_generator, filenames, vocab_path, hpm["vocab_size"], hpm["max_enc_len"], hpm["max_dec_len"], hpm["batch_size"], hpm["training"] )

	return dataset
