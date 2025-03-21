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
To perform reaction template prediction and compound generation with only the initial compound, run beam_search.py.
Input text should be written like
C c 1 c c c 2 [nH] c 3 c ( c 2 c 1 ) C N ( C ) C C 3
Please set src_test_path of config.py.

## translate.py
To generate compounds with a specified initial compound and reaction template, please run translate.py.
Input text should be written like
[749] C c 1 c c c 2 [nH] c 3 c ( c 2 c 1 ) C N ( C ) C C 3
Please set src_test_path of config.py.