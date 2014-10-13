import sys
import numpy

from read_write import *
from ranking import *

def compose_vectors(word_context, word_vecs, context_vecs):
  vec_len = len(context_vecs[0])
  res = numpy.zeros(vec_len)
  for index, con_word in word_context.iteritems():
    res += numpy.multiply(word_vecs[con_word], context_vecs[index])
  res = numpy.tanh(res)
  return res

if __name__=='__main__':

  eval_dataset, context_vec_file, word_vec_file = (sys.argv[1], sys.argv[2],
                                                   sys.argv[3])

  # Read the context vectors
  context_vecs = {}
  for line_index, line in enumerate(open(context_vec_file, 'r')):
    if line_index == 0: continue
    vec_stuff = line.split()
    word_index, vec_len = (int(vec_stuff[0]), int(vec_stuff[1]))
    context_vecs[word_index] = list_to_vector_no_norm(vec_stuff[2:])  
  
  # Read the word vectors
  word_vecs = read_word_vectors_no_norm(word_vec_file)
 
  # Compute word similarities in context
  manual_dict, auto_dict = ({}, {})
  pair_num, not_found, total = (0, 0, 0)
  for line in open(eval_dataset, 'r'):
    total += 1
    line = line.strip().lower()
    word1, word2, con1, con2, rating = line.split("\t")
    if not (word1 in word_vecs and word2 in word_vecs):
      not_found += 1
      continue
    # Get the context of the wtarget words
    con1words, con2words = (con1.split(), con2.split())
    word1index, word2index = (con1words.index("<b>")+1, con2words.index("<b>")+1)
    word1context, word2context = ({}, {})
    for i in range(-6, -1):
      if i+word1index >= 0 and con1words[i+word1index] in word_vecs:
        word1context[i+1] = con1words[i+word1index]
      if i+word1index >= 0 and con2words[i+word2index] in word_vecs:
        word2context[i+1] = con2words[i+word2index]
    for i in range(2, 7):
      if i+word1index < len(con1words) and con1words[i+word1index] in word_vecs:
        word1context[i-1] = con1words[i+word1index]
      if i+word1index < len(con2words) and con2words[i+word2index] in word_vecs:
        word2context[i-1] = con2words[i+word2index]
    word1context[0] = word1
    word2context[0] = word2
    # Compose the vectors together now
    word1_con_vec = compose_vectors(word1context, word_vecs, context_vecs)
    word2_con_vec = compose_vectors(word2context, word_vecs, context_vecs)
    manual_dict[pair_num] = float(rating)
    auto_dict[pair_num] = cosine_sim(word1_con_vec, word2_con_vec)
    pair_num += 1
 
  print "Total pairs:", total
  print "Not found:", not_found
  print "Correlation:", spearmans_rho(assign_ranks(manual_dict),
                                      assign_ranks(auto_dict))
