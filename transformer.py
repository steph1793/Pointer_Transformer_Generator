import tensorflow as tf
from layers import Embedding, EncoderLayer, DecoderLayer
from utils import _calc_final_dist


class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				rate=0.1):
		super(Encoder, self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		
		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
							for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)
	    
	def call(self, x, training, mask):
		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)

		return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
				rate=0.1):
		super(Decoder, self).__init__()
		self.d_model = d_model
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.depth = d_model // self.num_heads
		self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
							for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)
		self.Wh = tf.keras.layers.Dense(1)
		self.Ws = tf.keras.layers.Dense(1)
		self.Wx = tf.keras.layers.Dense(1)
		self.V = tf.keras.layers.Dense(1)


	def call(self, embed_x, enc_output, training, look_ahead_mask, padding_mask):

		attention_weights = {}
		out = self.dropout(embed_x, training=training)

		for i in range(self.num_layers):
			out, block1, block2 = self.dec_layers[i](out, enc_output, training,
													look_ahead_mask, padding_mask)
		
			attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
			attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

		# out.shape == (batch_size, target_seq_len, d_model)



		#context vectors
		enc_out_shape = tf.shape(enc_output)
		context = tf.reshape(enc_output,(enc_out_shape[0], enc_out_shape[1], self.num_heads, self.depth) ) # shape : (batch_size, input_seq_len, num_heads, depth)
		context = tf.transpose(context, [0,2,1,3]) # (batch_size, num_heads, input_seq_len, depth)
		context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)

		attn = tf.expand_dims(block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)

		context = context * attn # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
		context = tf.reduce_sum(context, axis=3) # (batch_size, num_heads, target_seq_len, depth)
		context = tf.transpose(context, [0,2,1,3]) # (batch_size, target_seq_len, num_heads, depth)
		context = tf.reshape(context, (tf.shape(context)[0], tf.shape(context)[1], self.d_model)) # (batch_size, target_seq_len, d_model)

		# P_gens computing
		a = self.Wx(embed_x)
		b = self.Ws(out)
		c = self.Wh(context)
		p_gens = tf.sigmoid(self.V(a + b + c))

		return out, attention_weights,  p_gens


class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,batch_size, rate=0.1):
		super(Transformer, self).__init__()

		self.num_layers =num_layers
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.model_depth = d_model
		self.num_heads = num_heads

		self.embedding = Embedding(vocab_size, d_model)
		self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
		self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
		self.final_layer = tf.keras.layers.Dense(vocab_size)


	def call(self, inp, extended_inp,max_oov_len, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

		embed_x = self.embedding(inp)
		embed_dec = self.embedding(tar)

		enc_output = self.encoder(embed_x, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

		# dec_output.shape == (batch_size, tar_seq_len, d_model)
		dec_output, attention_weights, p_gens = self.decoder(embed_dec, enc_output, training, look_ahead_mask, dec_padding_mask)

		output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
		output = tf.nn.softmax(output) # (batch_size, tar_seq_len, vocab_size)
		#output = tf.concat([output, tf.zeros((tf.shape(output)[0], tf.shape(output)[1], max_oov_len))], axis=-1) # (batch_size, targ_seq_len, vocab_size+max_oov_len)

		attn_dists = attention_weights['decoder_layer{}_block2'.format(self.num_layers)] # (batch_size,num_heads, targ_seq_len, inp_seq_len)
		attn_dists = tf.reduce_sum(attn_dists, axis=1)/self.num_heads # (batch_size, targ_seq_len, inp_seq_len)


		final_dists =  _calc_final_dist( extended_inp, tf.unstack(output, axis=1) , tf.unstack(attn_dists, axis=1), tf.unstack(p_gens, axis=1), max_oov_len, self.vocab_size, self.batch_size)
		final_output =tf.stack(final_dists, axis=1)

		return final_output, attention_weights