

class Vocab:

	SENTENCE_START  = '<s>'
	SENTENCE_END = '</s>'

	PAD_TOKEN = '[PAD]'
	UNKNOWN_TOKEN = '[UNK]'
	START_DECODING = '[START]'
	STOP_DECODING = '[STOP]'

	def __init__(self, vocab_file, max_size):

		self.word2id = {Vocab.UNKNOWN_TOKEN : 0, Vocab.PAD_TOKEN : 1, Vocab.START_DECODING : 2, Vocab.STOP_DECODING : 3}
		self.id2word = {0 : Vocab.UNKNOWN_TOKEN, 1 : Vocab.PAD_TOKEN, 2 : Vocab.START_DECODING, 3 : Vocab.STOP_DECODING}
		self.count = 4

		with open(vocab_file, 'r') as f:
			for line in f:
				pieces = line.split()
				if len(pieces) != 2 :
					print('Warning : incorrectly formatted line in vocabulary file : %s\n' % line)
					continue

				w = pieces[0]
				if w in [Vocab.SENTENCE_START, Vocab.SENTENCE_END, Vocab.UNKNOWN_TOKEN, Vocab.PAD_TOKEN, Vocab.START_DECODING, Vocab.STOP_DECODING]:
					raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)

				if w in self.word2id:
					raise Exception('Duplicated word in vocabulary file: %s' % w)

				self.word2id[w] = self.count
				self.id2word[self.count] = w
				self.count += 1
				if max_size != 0 and self.count >= max_size:
					print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
					break

		print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self.count, self.id2word[self.count-1]))


	def word_to_id(self, word):
		if word not in self.word2id:
			return self.word2id[Vocab.UNKNOWN_TOKEN]
		return self.word2id[word]

	def id_to_word(self, word_id):
		if word_id not in self.id2word:
			raise ValueError('Id not found in vocab: %d' % word_id)
		return self.id2word[word_id]

	def size(self):
		return self.count


class Data_Helper:
	def article_to_ids(article_words, vocab):
		ids = []
		oovs = []
		unk_id = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
		for w in article_words:
			i = vocab.word_to_id(w)
			if i == unk_id: # If w is OOV
				if w not in oovs: # Add to list of OOVs
					oovs.append(w)
				oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
				ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
			else:
				ids.append(i)
		return ids, oovs


	def abstract_to_ids(abstract_words, vocab, article_oovs):
		ids = []
		unk_id = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
		for w in abstract_words:
			i = vocab.word_to_id(w)
			if i == unk_id: # If w is an OOV word
				if w in article_oovs: # If w is an in-article OOV
					vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
					ids.append(vocab_idx)
				else: # If w is an out-of-article OOV
					ids.append(unk_id) # Map to the UNK token id
			else:
				ids.append(i)
		return ids



	def output_to_words(id_list, vocab, article_oovs):
		words = []
		for i in id_list:
			try:
				w = vocab.id_to_word(i) # might be [UNK]
			except ValueError as e: # w is OOV
				assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
				article_oov_idx = i - vocab.size()
				try:
					w = article_oovs[article_oov_idx]
				except ValueError as e: # i doesn't correspond to an article oov
					raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
			words.append(w)
		return words



	def abstract_to_sents(abstract):
		"""Splits abstract text from datafile into list of sentences.
			Args:
			  abstract: string containing <s> and </s> tags for starts and ends of sentences
			Returns:
			  sents: List of sentence strings (no tags)
		"""
		cur = 0
		sents = []
		while True:
			try:
				start_p = abstract.index(Vocab.SENTENCE_START, cur)
				end_p = abstract.index(Vocab.SENTENCE_END, start_p + 1)
				cur = end_p + len(Vocab.SENTENCE_END)
				sents.append(abstract[start_p+len(Vocab.SENTENCE_START):end_p])
			except ValueError as e: # no more sentences
				return sents

	def get_dec_inp_targ_seqs( sequence, max_len, start_id, stop_id):
		"""
			Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
			Args:
				sequence: List of ids (integers)
				max_len: integer
				start_id: integer
				stop_id: integer
			Returns:
				inp: sequence length <=max_len starting with start_id
				target: sequence same length as input, ending with stop_id only if there was no truncation
		"""
		inp = [start_id] + sequence[:]
		target = sequence[:]
		if len(inp) > max_len: # truncate
			inp = inp[:max_len]
			target = target[:max_len] # no end_token
		else: # no truncation
			target.append(stop_id) # end token
		assert len(inp) == len(target)
		return inp, target
