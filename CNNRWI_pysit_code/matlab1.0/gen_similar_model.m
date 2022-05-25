function [ vp_cluster ] = gen_similar_model( vp_mig, kmean_num_first,kmean_num_second,max_iteration,first_level,second_level,vel_sigma)
%GEN_SIMILAR_MODEL Summary of this function goes here
%   Detailed explanation goes here
    disp('Input a 2-D model output a 2-D model only')
    [nz nx] = size(vp_mig);
    vp_mig = reshape(vp_mig,nz*nx,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1. Get 1st-level segmentation index map (value is from 1 to kmean_num_first)
    vp_index_map = kmeans_index_map(vp_mig,kmean_num_first,max_iteration)*first_level;%*first_level for 1st-level index number

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %2. Get 2nd-level segmentation index map (value is from 1*10000 to specific k-mean number*10000)
    for k = 1*first_level:first_level:kmean_num_first*first_level
%         disp(['Process ' num2str(k) '-th sub group'])
        %get binary map for index k
        vp_binary_mask = binary_segmentation(vp_index_map,k);
        %extract initial model at this k group and set other parts to be 0
        vp_mig_k_group = vp_mig .* vp_binary_mask;
        %add index numbers to each sub-zone group, kmean_num_second may be
        %reduced if kmean_num_second cannot be found in this group
        vp_index_map_k = k + kmeans_index_map(vp_mig_k_group,kmean_num_second,max_iteration)*second_level;%+1 becasue the background is also a group
        for i = 1:length(vp_binary_mask)
            if vp_binary_mask(i) == 1 %make sure only add new sub index to specific group
                vp_index_map(i) = vp_index_map_k(i);
            end
        end
        figure(1)
        vp_binary_mask2d = reshape(vp_binary_mask,nz,nx);
        imagesc(vp_binary_mask2d)
        drawnow;pause(.5)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %3. Get each group's mean and variance
    %   Notice: This hash map may have empty value, hash-map value is the number of points
    hash_map_count = zeros(kmean_num_first*first_level+kmean_num_second*second_level,1);
    hash_map_value = zeros(kmean_num_first*first_level+kmean_num_second*second_level,2);%mean,variance

    vp_mig = reshape(vp_mig,nz*nx,1);
    % compute mean values for each group
    for i = 1:nz*nx
        hash_map_count(vp_index_map(i)) = hash_map_count(vp_index_map(i)) + 1;
        hash_map_value(vp_index_map(i),1) = hash_map_value(vp_index_map(i),1) +vp_mig(i);
    end
    hash_map_value(:,1) = hash_map_value(:,1) ./ hash_map_count;

    % compute variance for each group
    % vars(x) = 1/n*sum(x - x_bar)^2
    for i = 1:nz*nx
        hash_map_value(vp_index_map(i),2) = hash_map_value(vp_index_map(i),2) + (vp_mig(i) - hash_map_value(vp_index_map(i),1))^2;
    end
    hash_map_value(:,2) = hash_map_value(:,2) ./ hash_map_count;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %4. Randomly pick a value for each clusted group as  velocity value
    %get random value for each group which satisfies normal distribution
    hash_map_value_randn = normal_dist(hash_map_value(:,1),hash_map_value(:,2)*vel_sigma);
    vp_cluster = zeros(nz,nx);
    vp_index_map = reshape(vp_index_map,nz,nx);
    for ix = 1:nx
        for iz = 1:nz
            vp_cluster(iz,ix) = hash_map_value_randn(vp_index_map(iz,ix),1);
        end
    end
end

