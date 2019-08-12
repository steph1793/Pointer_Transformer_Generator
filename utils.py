
import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates

def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
							np.arange(d_model)[np.newaxis, :],
							d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding, dtype=tf.float32)



def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 1), tf.float32)

	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
	# Encoder padding mask
	enc_padding_mask = create_padding_mask(inp)

	# Used in the 2nd attention block in the decoder.
	# This padding mask is used to mask the encoder outputs.
	dec_padding_mask = create_padding_mask(inp)

	# Used in the 1st attention block in the decoder.
	# It is used to pad and mask future tokens in the input received by 
	# the decoder.
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

	return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead) 
	but it must be broadcastable for addition.

	Args:
	q: query shape == (..., seq_len_q, depth)
	k: key shape == (..., seq_len_k, depth)
	v: value shape == (..., seq_len_v, depth_v)
	mask: Float tensor with shape broadcastable 
	      to (..., seq_len_q, seq_len_k). Defaults to None.

	Returns:
	output, attention_weights
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)  

	# softmax is normalized on the last axis (seq_len_k) so that the scores
	# add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

	return output, attention_weights

def _calc_final_dist( _enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
	"""Calculate the final distribution, for the pointer-generator model

	Args:
	vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
	attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

	Returns:
	final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
	"""
	# Multiply vocab dists by p_gen and attention dists by (1-p_gen)
	vocab_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
	attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, attn_dists)]

	# Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
	extended_vsize = vocab_size + batch_oov_len # the maximum (over the batch) size of the extended vocabulary
	extra_zeros = tf.zeros((batch_size, batch_oov_len ))
	vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

	# Project the values in the attention distributions onto the appropriate entries in the final distributions
	# This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
	# This is done for each decoder timestep.
	# This is fiddly; we use tf.scatter_nd to do the projection
	batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
	batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
	attn_len = tf.shape(_enc_batch_extend_vocab)[1] # number of states we attend over
	batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
	indices = tf.stack( (batch_nums, _enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
	shape = [batch_size, extended_vsize]
	attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

	# Add the vocab distributions and the copy distributions together to get the final distributions
	# final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
	# Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
	final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

	return final_dists