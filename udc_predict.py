import time
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
from models.dual_encoder import dual_encoder_model
import pandas as pd
from termcolor import colored
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load data for predict
# test_df = pd.read_csv("./data/test.csv")
# elementId = 4
# INPUT_CONTEXT = test_df.Context[elementId]
# POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

INPUT_CONTEXT = "How do I view installed program"
# POTENTIAL_RESPONSES = np.asarray([ "grep packages","Can't be done","Click the ubuntu button on top of the shortcut bar"])
train_df = pd.read_csv("./data/train.csv")
INPUT_CONTEXT = "Firefox having issues when resuming from suspend."

tokens = word_tokenize(INPUT_CONTEXT)
stemmer = SnowballStemmer("english")
INPUT_CONTEXT=""
for token in tokens:
    INPUT_CONTEXT = INPUT_CONTEXT + " " + stemmer.stem(token)
print('####################################')
print(INPUT_CONTEXT)
print('####################################')
for i in range(10):
    POTENTIAL_RESPONSES = np.asarray(train_df["Utterance"][i*8000:8000*(i+1)].tolist())

    def get_features(context, utterances):
      context_matrix = np.array(list(vp.transform([context])))
      utterance_matrix = np.array(list(vp.transform([utterances[0]])))
      context_len = len(context.split(" "))
      utterance_len = len(utterances[0].split(" "))
      features =  {
            "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
            "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
            "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
            "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
            "len":len(utterances)
      }

      for i in range(1,len(utterances)):
          utterance = utterances[i];

          utterance_matrix = np.array(list(vp.transform([utterance])))
          utterance_len = len(utterance.split(" "))

          features["utterance_{}".format(i)] = tf.convert_to_tensor(utterance_matrix, dtype=tf.int64)
          features["utterance_{}_len".format    (i)] = tf.constant(utterance_len, shape=[1,1], dtype=tf.int64)

      return features, None

    if __name__ == "__main__":
      # tf.logging.set_verbosity(tf.logging.INFO)
      hparams = udc_hparams.create_hparams()
      model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)

      estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

      starttime = time.time()

      if float(tf.__version__[0:4])<0.12: #check TF version to select method
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES),as_iterable=True)
      else:
          prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES))
      results = next(prob)

      endtime = time.time()

      print('\n')
      print(colored('[Predict time]', on_color='on_red',color="white"),"%.2f sec" % round(endtime - starttime,2))
      print(colored('[     Context]', on_color='on_blue',color="white"),INPUT_CONTEXT)
      # print("[Results value ]",results)
      answerId = results.argmax(axis=0)
      print (colored('[      Answer]', on_color='on_green'),POTENTIAL_RESPONSES[answerId])