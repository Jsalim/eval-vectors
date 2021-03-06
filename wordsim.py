import sys

from read_write import read_word_vectors
from ranking import *

if __name__=='__main__':
    
  wordSimFile = sys.argv[1]
  wordVectorFile = sys.argv[2]
  wordVectors = read_word_vectors(wordVectorFile)
    
  manualDict, autoDict = ({}, {})
  notFound = 0
  for line in open(wordSimFile,'r'):
    line = line.strip().lower()
    word1, word2, val = line.split()
    if word1 in wordVectors and word2 in wordVectors:
      manualDict[(word1, word2)] = float(val)
      autoDict[(word1, word2)] = cosine_sim(wordVectors[word1], wordVectors[word2])
    else:
      notFound += 1
  print
  print 'No. pairs not found: ', notFound
  print "rho = ", spearmans_rho(assign_ranks(manualDict), assign_ranks(autoDict)) 
