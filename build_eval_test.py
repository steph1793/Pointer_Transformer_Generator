import tensorflow as tf
from tensorflow.python.training import training_util
from training_helper import train_model
from batcher import batcher
from transformer import Transformer



def my_model(features, labels, mode, params):
	transformer = Transformer(
		num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], 
		vocab_size=params["vocab_size"], batch_size=params["batch_size"])


	assert mode == tf.estimator.ModeKeys.TRAIN

	increment_step = training_util._increment_global_step(1)
	train_op, loss = train_model(features, labels, params, transformer )
	tf.summary.scalar("loss", loss)
	print(transformer.summary())

	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group([train_op, increment_step]))



def build_model(params):

	config = tf.estimator.RunConfig(
		tf_random_seed=params["seed"], 
		log_step_count_steps=params["log_step_count_steps"],
		save_summary_steps=params["save_summary_steps"]
	)

	return tf.estimator.Estimator(
					model_fn=my_model,
					params=params, config=config, model_dir=params["model_dir"] )


def train(model, params):
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	logdir = "{}/logdir".format(params["model_dir"])

	model.train(input_fn = lambda : batcher(params["data_dir"], params["vocab_path"], params), 
				max_steps=params["max_steps"],
				hooks=[tf.estimator.CheckpointSaverHook(
					checkpoint_dir,
					save_steps=params["checkpoints_save_steps"]
				), tf.estimator.SummarySaverHook(save_steps=params["save_summary_steps"], output_dir=logdir, scaffold=tf.compat.v1.train.Scaffold())])



def eval(model, params):
	pass

def test(model, params):
	pass