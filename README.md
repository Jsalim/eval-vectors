#eval-vectors

A bunch of scripts for evaluating word vectors on word similarity tasks. These scripts can read word vectors in ".gz" format, so no need to decompress the big files.

###all-wordsim
```python all_wordsim.py word_vector_file```

###wordsim
```python wordsim.py word_sim_dataset word_vector_file```

###context-wordsim
Evaluating vectors on the contextual word similarity dataset (Huang et al, 2012)

```python wordsim.py word_sim_dataset word_vector_file```
