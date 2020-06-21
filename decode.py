# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os,re
import time
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
import logging
import numpy as np
import string    
from data import Vocab
from tensorflow.python import pywrap_tensorflow
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

FLAGS = tf.app.flags.FLAGS
#single_pass=True
SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab,hps):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config()) 
    self._hps=hps

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)
    

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    #if FLAGS.single_pass:
    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
      if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)


  def decode(self,my_embedding):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)
        return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
      decoded_output = ' '.join(decoded_words) # single string
      summarize_texted=self.summarize_texted(self._hps,original_article,decoded_output,self._vocab,my_embedding)

      if FLAGS.single_pass:
        self.write_for_rouge(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded
      else:
        print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
        self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()

  def summarize_texted(self,hps,original_article,decoded_output,vocab,my_embedding):
      ckpt_file_name ='/home/ddd/data/cnndailymail3/finished_files/exp_logs/train/model.ckpt-33831'#model name
      name_variable_to_restore='seq2seq/embedding/embedding'#variable name
      decoded_output_list=re.split(r'[.]', decoded_output.strip())
      top_n=len(decoded_output_list)
      original_article_my=original_article.strip(string.punctuation) 
      decoded_output_my=decoded_output.strip(string.punctuation) 
      decoded_output_my_str=original_article_my+'.'+decoded_output_my
      

      def read_article(self,decoded_output_my_str):
          sentence_word_list=[]
          sentences_a=re.split(r'[.]', decoded_output_my_str.strip())#split sentence with '.'
          for line in sentences_a:#line_split is a list include many str sentences
            line_seg=line.split() #many str word
            sentence_word_list.append(line_seg)
          return sentence_word_list,sentences_a
      def the_sentence_vectors(self,sentences, my_embedding,vocab):
          enc_input_list=[]
          for sent in sentences:
              enc_input=[vocab.word2id(w) for w in sent] 
              enc_input_list.append(enc_input)#a list in sentence id
          #sents_input_id = [vocab.word2id(w) for w in sents_words]
          # s1=tf.nn.embedding_lookup(my_embedding,sents_input_id)
          
          # sess.run(tf.variables_initializer([my_embedding], name='init'))#run variable
          sentence_vectors = []    #句子向量
          #a = tf.constant([10, 20])
          #b = tf.constant([1.0, 2.0])
          #asas=self._sess.run(a)
          self._sess.run(tf.variables_initializer([my_embedding], name='init'))#run variable

          #i_emb=tf.nn.embedding_lookup(my_embedding,36)
          #i_emb_value=self._sess.run(i_emb)#get value of i_emb
          
          #qqq=sess.run(tf.nn.embedding_lookup(my_embedding,1)
          for i in enc_input_list:
              if len(i):
                  # i_emb=tf.nn.embedding_lookup(my_embedding,i)
                  # i_emb_value=sess.run(i_emb)#get value of i_emb
                  
                  #v = sum([word_embeddings.get(w, np.random.uniform(0, 1, 128)) for w in i]) / (len(i) + 0.001)
                  v = sum([self._sess.run(tf.nn.embedding_lookup(my_embedding,w)) for w in i]) / (len(i) + 0.001)
                  ##v = sum([tf.nn.embedding_lookup(my_embedding,w) for w in i]) / (len(i) + 0.001)
              else:
                  v = np.random.uniform(0, 1, 128)
              sentence_vectors.append(v)

          return sentence_vectors

      def build_similarity_matrix(self, sentences, sentence_vectors):
          # Create an empty similarity matrix
          similarity_matrix = np.zeros((len(sentences), len(sentences)))

          for idx1 in range(len(sentences)):
             for idx2 in range(len(sentences)):
                 if idx1 != idx2:  # ignore if both are same sentences
                  #continue
                  # similarity_matrix[idx1][idx2] = self.sentence_similarity(sentence_vectors[idx1].reshape(1,1),sentence_vectors[idx2].reshape(1,1))
                  # similarity_matrix[idx1][idx2] = self.sentence_similarity(sentence_vectors[idx1],sentence_vectors[idx2])
                     similarity_matrix[idx1][idx2] = cosine_similarity(sentence_vectors[idx1].reshape(1,128),sentence_vectors[idx2].reshape(1,128))[0,0]
          return similarity_matrix

      def generate_summary(self, decoded_output_my_str,top_n,ckpt_file_name,name_variable_to_restore,vocab,my_embedding):
          summarize_text = []
         
          #reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file_name)#read .ckpt
          # var_to_shape_map = reader.get_variable_to_shape_map()#get variable
          # my_embedding = tf.get_variable("my_embedding", var_to_shape_map[name_variable_to_restore], trainable=False)#rename 'embedding'variable name to my_embedding
          #my_embedding = tf.get_variable('embedding', [50000, hps.emb_dim], dtype=tf.float32)
         
          # Step 1 - Read text anc split it
          sentences,line_split= read_article(self,decoded_output_my_str)
          sentence_vectors= the_sentence_vectors(self,sentences,my_embedding,vocab)
          # Step 2 - Generate Similary Martix across sentences
          sentence_similarity_martix = build_similarity_matrix(self,sentences, sentence_vectors)

          # Step 3 - Rank sentences in similarity martix
          sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
          scores = nx.pagerank(sentence_similarity_graph,max_iter=1000)#max_iter is 最大迭代次数

           # Step 4 - Sort the rank and pick top sentences
          ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(line_split)), reverse=True)
           # print("Indexes of top ranked_sentence order are ", ranked_sentence)

          for i in range(top_n):
              summarize_text.append("".join(ranked_sentence[i][1]))
          summarize_texted=". ".join(summarize_text)
          return summarize_texted,sentences
          # Step 5 - Offcourse, output the summarize texr
          # print("Summarize Text: \n", ". ".join(summarize_text))
          # print("lllaaaaaa\n",summarize_texted)
      
      summarize_texted,sentences=generate_summary(self,decoded_output_my_str,top_n,ckpt_file_name,name_variable_to_restore,vocab,my_embedding)
      return summarize_texted



  def write_for_rouge(self, reference_sents, decoded_words, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index(".")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print("---------------------------------------------------------------------------")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print("---------------------------------------------------------------------------")
  results_fname = os.path.join("/home/ddd/data/cnndailymail3/finished_files/exp_logs/decode", 'GENERATED_SUMMARY.txt')
  with open(results_fname,"w") as f1:
    f1.write(article)
    f1.write("------------------------------------------------")
    f1.write(abstract)
    f1.write("----------------------------")
    f1.write(decoded_output)


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155('/home/ddd/project/rouge_files/pyrouge/tools/ROUGE-1.5.5')
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate(rouge_args='-e {}/data -a -2 -1 -c 95 -U -n 2 -w 1.2 -b 75'.format('/home/ddd/project/rouge_files/pyrouge/tools/ROUGE-1.5.5'))
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname
