%% load dataset
load('./blogcatalog.mat');

%% Step I: random walk
num_walk = 10;
len_walk = 80;
num_node = size(network,1);
[indy,indx] = find(network');
[temp,m,n] = unique(indx);
node_list_len = [m(2:end);length(indx)+1] - m; % sum(counts)
node_list = mat2cell(indy,node_list_len);
% neg_sam_table = int64([1:num_node]'); uniform sampling for negative data

% let's have a walk
walks = zeros(num_walk*num_node,len_walk,'int64');
for ww=1:num_walk
    for ii=1:num_node
        seq = zeros(1,len_walk);
        seq(1) = ii;
        current_e = ii;
        for jj=1:len_walk-1
            rand_ind = randi([1 node_list_len(seq(jj))],1);
            seq(jj+1) = node_list{seq(jj)}(rand_ind,:);
        end
        walks(ii+(ww-1)*num_node,:) = seq;
    end
end

% generate negative sample table
[r,~] = find(network);
tab_degree = tabulate(r);
freq = tab_degree(:,3).^(0.75);
neg_sam_table = int64(repelem(tab_degree(:,1),round(10000000* freq/sum(freq)))); % unigram with 0.75 power



%% Step II: learn node embeddings
% learn! learn! learn!
dim_emb = 128;
num_pos_sample = 20;
learning_rate = 0.025;
embs_ini = (rand(num_node,dim_emb)-0.5)/dim_emb; 
embs_len = sqrt(sum(embs.^(2), 2));
embs_ini = embs_ini./(repmat(embs_len, 1, dim_emb));
beta = 0.4;
K_neg = 10;
num_threads = 4;
order = 5; % must be smaller than len_walk
norm_flag = 1;

% mex embRWGS.c       
tic;
[embs,~] = embRWGS(walks',embs_ini',num_pos_sample,learning_rate,K_neg,neg_sam_table,num_threads,beta,order, norm_flag);
embs = embs';
toc;

% save embeddings
save('./embeddings_RWGS.mat','embs');












