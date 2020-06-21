#model name is model.ckpt-33831,
#file_i_name='/home/ddd/data/cnndailymail3/train_my_emb.txt'
#generay the abstract in terminal
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from data import Vocab 
import data
import re
import numpy as np
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
file_name ='/home/ddd/data/cnndailymail3/finished_files/exp_logs/train/model.ckpt-33831'#model name
name_variable_to_restore='seq2seq/embedding/embedding'#variable name
vocab_path="/home/ddd/data/cnndailymail3/finished_files/vocab"
class generatesumm:

    def read_article(self,file_i_name):
       
        sentence_word_list=[]
        with open (file_i_name,'r',encoding='utf-8') as f1:
            sentences=f1.readlines()
            for sen in sentences:
                sentences_a=re.split(r'[.]', sen.strip())#split sentence with '.'
       
        for line in sentences_a:#line_split is a list include many str sentences
            line_seg=line.split() #many str word
            sentence_word_list.append(line_seg)
        return sentence_word_list,sentences_a#sentence_word_list is a list with list sentences in str word
  
    def sentence_vectors(self,sentences, my_embedding,vocab ):
        enc_input_list=[]
        for sent in sentences:
            enc_input=[vocab.word2id(w) for w in sent] 
            enc_input_list.append(enc_input)#a list in sentence id
        #sents_input_id = [vocab.word2id(w) for w in sents_words]
       # s1=tf.nn.embedding_lookup(my_embedding,sents_input_id)
        sess = tf.Session()   #create seesion
        sess.run(tf.variables_initializer([my_embedding], name='init'))#run variable
        sentence_vectors = []    #句子向量
        for i in enc_input_list:
            if len(i):
               # i_emb=tf.nn.embedding_lookup(my_embedding,i)
               # i_emb_value=sess.run(i_emb)#get value of i_emb
                
                #v = sum([word_embeddings.get(w, np.random.uniform(0, 1, 128)) for w in i]) / (len(i) + 0.001)
                v = sum([sess.run(tf.nn.embedding_lookup(my_embedding,w)) for w in i]) / (len(i) + 0.001)
              
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

    def generate_summary(self, file_i_name,top_n):
        summarize_text = []
        vocab = Vocab(vocab_path,50000) #create vocab
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)#read .ckpt
        var_to_shape_map = reader.get_variable_to_shape_map()#get variable
        my_embedding = tf.get_variable("my_embedding", var_to_shape_map[name_variable_to_restore], trainable=False)#rename 'embedding'variable name to my_embedding
        

    # Step 1 - Read text anc split it
        sentences,line_split= self.read_article(file_i_name)
        sentence_vectors=self.sentence_vectors(sentences,my_embedding,vocab)
    # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.build_similarity_matrix(sentences, sentence_vectors)

    # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph,max_iter=300)#max_iter is 最大迭代次数

    # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(line_split)), reverse=True)
       # print("Indexes of top ranked_sentence order are ", ranked_sentence)

        for i in range(top_n):
            summarize_text.append("".join(ranked_sentence[i][1]))
        summarize_texted=". ".join(summarize_text)
        #return summarize_texted,sentences
    # Step 5 - Offcourse, output the summarize texr
        print("Summarize Text: \n", ". ".join(summarize_text))
        print("lllaaaaaa\n",summarize_texted)

    # let's begin

if __name__ == '__main__':
    file_i_name='/home/ddd/data/cnndailymail3/train_my_emb.txt'
    ss=generatesumm()
    ss.generate_summary(file_i_name,3)