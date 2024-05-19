## transformer_train.py
Training of Transformer model.

## GCN_train.py
Training of GCN model.

## mcts.py
Molecular generation using MCTS. 

## preprocess.py
(optional) Making vocabulary.
In transformer_train.py, the model creates vocabulary file from training dataset.

## beam_search.py
Translation of input file. Please set src_test_path of config.py.
Input text should be written like
C c 1 c c c 2 [nH] c3 c ( c 2 c 1 ) C N ( C ) C C 3
Please set file name of output.

## translate.py
If you add reaction template manually, please set input text like 
[749] C c 1 c c c 2 [nH] c3 c ( c 2 c 1 ) C N ( C ) C C 3
and use translate.py