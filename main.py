import tensorflow as tf
import argparse
from build_eval_test import build_model, train
import os

def main():

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length")
	
	parser.add_argument("--max_dec_len", default=100, help="Decoder input max sequence length")
	
	parser.add_argument("--batch_size", default=16, help="batch size")
	
	parser.add_argument("--vocab_size", default=50000, help="Vocabulary size")
	
	parser.add_argument("--num_layers", default=3, help="Model encoder and decoder number of layers")
	
	parser.add_argument("--model_depth", default=512, help="Model Embedding size")
	
	parser.add_argument("--num_heads", default=8, help="Multi Attention number of heads")
	
	parser.add_argument("--dff", default=2048, help="Dff")

	parser.add_argument("--seed", default=123, help="Seed")
	
	parser.add_argument("--log_step_count_steps", default=1, help="Log each N steps")
	
	parser.add_argument("--max_steps",default=230000, help="Max steps for training")
		
	parser.add_argument("--save_summary_steps", default=10000, help="Save summaries every N steps")
	
	parser.add_argument("--checkpoints_save_steps", default=10000, help="Save checkpoints every N steps")
	
	parser.add_argument("--mode", help="training, eval or test options")

	parser.add_argument("--model_dir", help="Model folder")

	parser.add_argument("--data_dir",  help="Data Folder")

	parser.add_argument("--vocab_path", help="Vocab path")

	
	args = parser.parse_args()
	params = vars(args)

	assert params["mode"], "mode is required. train, test or eval option"
	if params["mode"] == "train":
		params["training"] = True ; params["eval"] = False ; params["test"] = False
	elif params["mode"] == "eval":
		params["training"] = False ; params["eval"] = True ; params["test"] = False
	elif params["mode"] == "test":
		params["training"] = False ; params["eval"] = False ; params["test"] = True;
	else:
		raise NameError("The mode must be train , test or eval")
	assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
	assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"



	model = build_model(params)

	if params["training"]:
		train(model, params)
	elif params["eval"]:
		pass
	elif params["test"]:
		pass


if __name__ == "__main__":
	main()