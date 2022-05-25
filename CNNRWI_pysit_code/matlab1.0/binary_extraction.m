function [ binary_set ] = binary_extraction( orig_figure,kmean_num,null_num)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    %extract two sub figures (ignore the blank area)
    
    %null_num = 1 if orig_figure contains blank area which will be counted
    %as a cluster but should not be considered as a cluster in this binary
    %extraction
    max_iteration = 1000; % This is  USED since singleton will sometimes decrease the number of clusters!!!
    vp_index_map = kmeans_index_map(orig_figure,kmean_num+null_num,max_iteration);%*first_level for 1st-level index number
    binary_set = zeros(length(orig_figure),kmean_num);
    count = 0;
    for k = 1:kmean_num+null_num
%         disp(['Process ' num2str(k) '-th sub group'])
        %get binary map for index k
        vp_binary_mask = binary_segmentation(vp_index_map,k);
        
        %avoid save the removed part as feature:
        if sum(vp_binary_mask.*orig_figure) == 0
            continue;
        end
        count = count + 1;
        vp_mig_k_group = orig_figure .* vp_binary_mask;
        binary_set(:,count) = vp_mig_k_group;
    end

end

