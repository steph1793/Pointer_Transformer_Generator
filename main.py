import tensorflow as tf
import argparse
from build_eval_test import build_model, train, test
import os

def main():

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
	
	parser.add_argument("--max_dec_len", default=100, help="Decoder input max sequence length", type=int)
	
	parser.add_argument("--batch_size", default=16, help="batch size", type=int)
	
	parser.add_argument("--vocab_size", default=50000, help="Vocabulary size", type=int)
	
	parser.add_argument("--num_layers", default=3, help="Model encoder and decoder number of layers", type=int)
	
	parser.add_argument("--model_depth", default=512, help="Model Embedding size", type=int)
	
	parser.add_argument("--num_heads", default=8, help="Multi Attention number of heads", type=int)
	
	parser.add_argument("--dff", default=2048, help="Dff", type=int)

	parser.add_argument("--seed", default=123, help="Seed", type=int)
	
	parser.add_argument("--log_step_count_steps", default=1, help="Log each N steps", type=int)
	
	parser.add_argument("--max_steps",default=230000, help="Max steps for training", type=int)
		
	parser.add_argument("--save_summary_steps", default=10000, help="Save summaries every N steps", type=int)
	
	parser.add_argument("--checkpoints_save_steps", default=10000, help="Save checkpoints every N steps", type=int)
	
	parser.add_argument("--mode", help="training, eval or test options")

	parser.add_argument("--model_dir", help="Model folder")

	parser.add_argument("--data_dir",  help="Data Folder")

	parser.add_argument("--vocab_path", help="Vocab path")

	
	args = parser.parse_args()
	params = vars(args)
	print(params)

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



	if params["training"]:
		train( params)
	elif params["eval"]:
		pass
	elif not params["training"]:
		pass


if __name__ == "__main__":
	main()