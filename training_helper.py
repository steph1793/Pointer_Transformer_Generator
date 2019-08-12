import tensorflow as tf
from utils import create_masks

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)



def train_step(features, labels, params, model, optimizer, loss_object):
  
	enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input"], labels["dec_input"])

	with tf.GradientTape() as tape:
		output, attn_weights = model(features["enc_input"],features["extended_enc_input"], features["max_oov_len"], labels["dec_input"], training=params["training"], 
									enc_padding_mask=enc_padding_mask, 
									look_ahead_mask=combined_mask,
									dec_padding_mask=dec_padding_mask)
		loss = loss_function(loss_object, labels["dec_target"], output)

	gradients = tape.gradient(loss, model.trainable_variables)    
	return optimizer.apply_gradients(zip(gradients, model.trainable_variables)), loss


def train_model(features, labels, params, model):
	learning_rate = CustomSchedule(params["model_depth"])
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

	return train_step(features, labels, params, model, optimizer, loss_object)

