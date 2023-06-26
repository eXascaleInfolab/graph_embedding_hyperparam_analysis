This is the implementation used in our paper of graph embedding hyperparameter analysis. 
Dingqi Yang, Bingqing Qu, Rana Hussein, Paolo Rosso, Philippe Cudre-Mauroux, and Jie Liu, Revisiting Embedding Based Graph Analyses: Hyperparameters Matter! IEEE Transactions on Knowledge and Data Engineering (TKDE), 2023.

It contains two types of algorithms:
- Factorization-based graph embedding techniques
- Random-walk graph-sampling based techniques

How to use (Tested on MATLAB 2017a and 2017b):
- embMF:
1. run experiment_MF.m


- embRWGS:
1. Compile embRWGS.c using mex: mex embRWGS.c
2. Run experiment_RWGS.m


- evaluation on the node classification task (using Deepwalk testing code):
1. run evaluation_node_classification.m
or from command line:
1. python ./scoring.py ./blogcatalog.mat ./embeddings_MF.mat ./classification_res_MF.mat


Please cite our paper if you publish material using this code.

