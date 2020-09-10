function [embs] = hune_with_options(network, prox_metric, sppmi, fac_method, dim_emb, alpha_Katz, order, K_shifted, norm_flag)
% HUNE: factorization-based graph embedding

switch prox_metric
    case 'katz'
        if isinf(order)
            sp_radius = abs(eigs(network,1));
            weight = alpha_Katz/sp_radius;
            prox_mat = (inv(eye(size(network))-weight*network) - eye(size(network)))./weight;
            
        else
            P_temp = network;
            prox_mat = network;
            for ii=2:order
                P_temp = alpha_Katz*P_temp*network;
                prox_mat = prox_mat + P_temp;
            end
            weight = sum(alpha_Katz.^[0:1:order-1]);
            prox_mat = prox_mat./weight;
            
        end
        
        
    case 'rand_walk'
        if isinf(order)
            P = network.*(1./sum(network,2));
            P_sum_inf = inv(eye(size(P))-alpha_Katz*P) - eye(size(P));
            weight = alpha_Katz/(1-alpha_Katz);
            prox_mat = sum(network(:))/weight*(P_sum_inf.*(1./sum(network,1)));
            
        else
            P = network.*(1./sum(network,2));
            P_temp = P;
            P_sum = P;
            for ii=2:order
                P_temp = alpha_Katz*P_temp*P;
                P_sum = P_sum + P_temp;
            end
            weight = sum(alpha_Katz.^[0:1:order-1]);
            prox_mat = sum(network(:))/weight*(P_sum.*(1./sum(network,1)));
        end
end



switch sppmi
    case 'sppmiTrue'
        A = max(log(prox_mat)-log(K_shifted),0);
    case 'sppmiFalse'
        A = prox_mat;
end


if(length(find(A))==0)
    embs = NaN;
    disp('K_shifted is set too high, SPPMI matrix are all zeros!');
else
    switch fac_method
        case 'evd'
            [V_emb,D_emb] = eigs(A,dim_emb,'la');
            
            embs = V_emb*sqrt(D_emb);
            if sum(isnan(embs(:)))>0 || ~isreal(embs)
                disp('EVD1: A is singular or #postive eigenvalues are less than embedding dimension!');
            end
            
            
        case 'svd'
            [U,S,V] = svds(A,dim_emb);
            
            embs = U*sqrt(S);
            if sum(isnan(embs(:)))>0 || ~isreal(embs)
                embs = NaN;
                disp('SVD1: A is singular!');
            end  
    end
    
    
    if strcmp(norm_flag, 'norm')
        embs = embs./(repmat(sqrt(sum(embs.^(2), 2)), 1, dim_emb));
        if sum(isnan(embs(:)))>0 || ~isreal(embs)
            disp('Embs norm error! Some nodes have all-zeros embedding vectors.');
        end
    end
    
end






