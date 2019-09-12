import tensorflow as tf
from tensorflow.python.training import training_util
from training_helper import train_model
from predict_helper import predict
from batcher import batcher
from transformer import Transformer



def my_model(features, labels, mode, params):
	


	if mode == tf.estimator.ModeKeys.TRAIN:
		increment_step = training_util._increment_global_step(1)
		train_op, loss = train_model(features, labels, params, transformer )
		tf.summary.scalar("loss", loss)
		estimator_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group([train_op, increment_step]))
	

	elif mode == tf.estimator.ModeKeys.PREDICT :
		predictions, attn_weights = predict(features, params, transformer)
		estimator_spec = tf.estimator.EstimatorSpec(mode,  predictions={"predictions":predictions})
  
	print(transformer.summary())
	return estimator_spec


def build_model(params):

	config = tf.estimator.RunConfig(
		tf_random_seed=params["seed"], 
		log_step_count_steps=params["log_step_count_steps"],
		save_summary_steps=params["save_summary_steps"]
	)

	return tf.estimator.Estimator(
					model_fn=my_model,
					params=params, config=config, model_dir=params["model_dir"] )


def train(params):
	assert params["training"], "change training mode to true"

	tf.compat.v1.logging.info("Building the model ...")
	transformer = Transformer(
		num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], 
		vocab_size=params["vocab_size"], batch_size=params["batch_size"])


	tf.compat.v1.logging.info("Creating the batcher ...")
	b = batcher(params["data_dir"], params["vocab_path"], params)

	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	logdir = "{}/logdir".format(params["model_dir"])

	tf.compat.v1.logging.info("Starting the training ...")
	train_model(transformer, b, params)
	


def eval(model, params):
	pass


def test(model, params):
	assert not params["training"], "change training mode to false"
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	logdir = "{}/logdir".format(params["model_dir"])

	pred = model.predict(input_fn = lambda :  batcher(params["data_dir"], params["vocab_path"], params), 
		yield_single_examples=False)

	yield next(pred)