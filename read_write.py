import sys
import gzip
import numpy
import math

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):    
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')

  for lineNum, line in enumerate(fileObject):
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)      
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)        

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Read all the word vectors '''
def read_word_vectors_no_norm(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')

  for lineNum, line in enumerate(fileObject):
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

def list_to_vector(val_list):
  vector = []
  for i, val in enumerate(val_list):
    vector.append(float(val))
  vector = numpy.array(vector)
  vector /= math.sqrt((vector**2).sum() + 1e-6)
  return vector

def list_to_vector_no_norm(val_list):
  vector = []
  for i, val in enumerate(val_list):
    vector.append(float(val))
  vector = numpy.array(vector)
  return vector
