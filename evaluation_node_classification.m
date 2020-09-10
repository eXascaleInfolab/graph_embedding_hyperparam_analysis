% calssification evaluation using Deepwalk testing code: 
[status,cmdout] = system('python3 ./scoring.py ./blogcatalog.mat ./embeddings_MF.mat ./classification_res_MF.mat');
% [status,cmdout] = system('python3 ./scoring.py ./blogcatalog.mat ./embeddings_RWGS.mat ./classification_res_RWGS.mat');

% get results
load('./classification_res_MF.mat');
% load('./classification_res_RWGS.mat');
F1 = squeeze(mean(res,1));
disp(F1);

