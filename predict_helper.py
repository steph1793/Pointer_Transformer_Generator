import tensorflow as tf
from utils import create_masks

def predict(features, params, model):
  
  output = tf.tile([[2]], [params["batch_size"], 1]) # 2 = start_decoding
  
  for i in range(params["max_dec_len"]):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(features["enc_input"], output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = model(features["enc_input"],features["extended_enc_input"], features["max_oov_len"], output, training=params["training"], 
                               enc_padding_mask=enc_padding_mask, 
                               look_ahead_mask=combined_mask,
                               dec_padding_mask=dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return output, attention_weights