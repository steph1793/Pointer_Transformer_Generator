import tensorflow as tf
from utils import positional_encoding, scaled_dot_product_attention 

class Embedding(tf.keras.layers.Layer):

	def __init__(self, vocab_size, d_model):
		super(Embedding, self).__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model

		self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
		self.pos_encoding = positional_encoding(vocab_size, d_model)

	def call(self, x):
		embed_x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
		embed_x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		embed_x += self.pos_encoding[:, :tf.shape(x)[1], :]
		return embed_x



class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)

		self.dense = tf.keras.layers.Dense(d_model)

	def split_heads(self, x, batch_size):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention, 
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

		return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
	])


class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()

		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training, mask):

		attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

		return out2


class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderLayer, self).__init__()

		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)


	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		# enc_output.shape == (batch_size, input_seq_len, d_model)

		attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)

		attn2, attn_weights_block2 = self.mha2(
			enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

		ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

		return out3, attn_weights_block1, attn_weights_block2