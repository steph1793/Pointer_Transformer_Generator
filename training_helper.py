import tensorflow as tf
from utils import create_masks
import time

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


def train_step(features, labels, params, model, optimizer, loss_object, train_loss_metric):
  
	enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input"], labels["dec_input"])

	with tf.GradientTape() as tape:
		output, attn_weights = model(features["enc_input"],features["extended_enc_input"], features["max_oov_len"], labels["dec_input"], training=params["training"], 
									enc_padding_mask=enc_padding_mask, 
									look_ahead_mask=combined_mask,
									dec_padding_mask=dec_padding_mask)
		loss = loss_function(loss_object, labels["dec_target"], output)

	gradients = tape.gradient(loss, model.trainable_variables)    
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	train_loss_metric(loss)

def train_model(model, batcher, params, ckpt, ckpt_manager):
	learning_rate = CustomSchedule(params["model_depth"])
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
	train_loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")

	try:
		for batch in batcher:
			t0 = time.time()
			train_step(batch[0], batch[1], params, model, optimizer, loss_object, train_loss_metric)
			t1 = time.time()
			
			print("step {}, time : {}, loss: {}".format(int(ckpt.step), t1-t0, train_loss_metric.result()))
			if int(ckpt.step) % params["checkpoints_save_steps"] ==0 :
				ckpt_manager.save(checkpoint_number=int(ckpt.step))
				print("Saved checkpoint for step {}".format(int(ckpt.step)))
			ckpt.step.assign_add(1)
			
	except KeyboardInterrupt:
		ckpt_manager.save(int(ckpt.step))
		print("Saved checkpoint for step {}".format(int(ckpt.step)))