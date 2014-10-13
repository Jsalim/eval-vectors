import sys

from read_write import read_word_vectors
from findMatch import cosine_sim
from ranking import spearmans_rho
from ranking import assign_ranks

if __name__=='__main__':  
  wordVectorFile = sys.argv[1]
  wordVectors = read_word_vectors(wordVectorFile)
  print '================================================================================='
  print "%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho"
  print '================================================================================='
  DIR = '/usr1/corpora/usr0-corpora/word-sim/'
  FILES = ['EN-MC-30.txt', 'EN-MTurk-287.txt', 'EN-RG-65.txt', 'EN-RW-STANFORD.txt', 'EN-WS-353-ALL.txt', 'EN-WS-353-REL.txt', 'EN-WS-353-SIM.txt', 'EN-MEN-TR-3k.txt', 'EN-YP-130.txt', 'EN-MTurk-771.txt']

  for i, FILE in enumerate(FILES):
    manualDict, autoDict = ({}, {})
    notFound, totalSize = (0, 0)
    for line in open(DIR+FILE,'r'):
      line = line.strip().lower()
      word1, word2, val = line.split()
      if word1 in wordVectors and word2 in wordVectors:
        manualDict[(word1, word2)] = float(val)
        autoDict[(word1, word2)] = cosine_sim(wordVectors[word1], wordVectors[word2])
      else:
        notFound += 1
        totalSize += 1    
    print "%6s" % str(i+1), "%20s" % FILE, "%15s" % str(totalSize),
    print "%15s" % str(notFound),
    print "%15.4f" % spearmans_rho(assign_ranks(manualDict), assign_ranks(autoDict))
