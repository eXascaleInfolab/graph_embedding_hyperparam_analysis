%% HUNE: factorization based high-order unique node embedding
% load dataset
load('./blogcatalog.mat');

% proximity metric: 'katz' or 'rand_walk'
prox = 'rand_walk';
% sppmi transformation: 'sppmiTrue' or 'sppmiFalse'
sppmi = 'sppmiTrue';
% factorization method: 'svd' or 'evd'
fac = 'evd';
% order of proximity (k in the paper): integers or inf
order_k = 2;
% Weight decay parameter (\alpha in the paper): any positive real value
alpha = 0.1;
% Shifting parameter (\gamma in the paper): any positive real value
gamma = 5;
% dimension of embeddings: any positive integer
dim_emb = 128;
% normalization of embeddings: 'norm' or 'unnorm'
norm_flag = 'norm';

% learn embeddings
tic;
[embs] = embMF(network, prox, sppmi, fac, dim_emb, alpha, order_k, gamma, norm_flag);
toc;

% save embeddings
save('./embeddings_MF.mat','embs');






