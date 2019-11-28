# Pointer_Transformer_Generator tensorflow 2.0.0

For the abstractive summarization task, I wanted to experiment the transformer model. I recreated a transformer model (thanks to tensorflow transformer tutorial) and added a pointer module (have a look at this paper for more informations on the pointer generator network : https://arxiv.org/abs/1704.04368 ).

PS : I will add very soon a section explaining the integration of the pointer module in the transformer

Please follow the next steps to launch the project :

## Step 1 : The data

### Option 1 : Download the data
Download the data (chunk files format : tfrecords)
https://drive.google.com/open?id=1uHrMWd7Pbs_-DCl0eeMxePbxgmSce5LO

### Option 2 : Download raw data and process it
Use this project : 
https://github.com/steph1793/CNN-DailyMail-Bin-To-TFRecords

## Step 2 : launch the project : 

**python main.py --max_enc_len=400 \ <br>
--max_dec_len=100 \ <br>
--batch_size=16 \ <br>
--vocab_size=50000 \ <br>
--num_layers=3 \ <br>
--model_depth=512 \ <br>
--num_heads=8 \ <br>
--dff=2048 \ <br>
--seed=123 \ <br>
--log_step_count_steps=1 \ <br>
--max_steps=230000 \ <br>
--mode=train \ <br>
--save_summary_steps=10000 \ <br>
--checkpoints_save_steps=10000 \ <br>
--model_dir=model_folder \ <br>
--data_dir=data_folder \ <br>
--vocab_path=vocab \ <br>**

PS : Feel free to change some of the hyperparameters<br>
python main.py --help , for more details on the hyperparameters



## Requirements
- python >= 3.6
- tensorflow 2.0.0
- argparse
- os
- glob
- numpy

