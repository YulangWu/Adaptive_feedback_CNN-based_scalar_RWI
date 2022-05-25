function [ hash_map_value ] = get_mean_std( segmented_layer )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
    [totaln num_feature_layers] = size(segmented_layer);
    binary_segmented_layer = sign(segmented_layer);
    hash_map_count = zeros(num_feature_layers,1);
    hash_map_value = zeros(num_feature_layers,2);%mean,std
    for i = 1 : num_feature_layers
        hash_map_count(i,1) = sum(binary_segmented_layer(:,i)); %sum of 1's
        % compute mean for each group
        % mean(x) = 1/n*sum(x)
        if hash_map_count(i,1) ~= 0
            hash_map_value(i,1) = sum(segmented_layer(:,i))/hash_map_count(i,1);
            % compute std for each group
            % vars(x) = sqrt(1/n*sum(x - x_bar)^2)
            for j = 1:totaln
                if binary_segmented_layer(j,i) == 1
                    hash_map_value(i,2) = hash_map_value(i,2) + (segmented_layer(j,i)-hash_map_value(i,1))^2;
                end
            end
            hash_map_value(i,2) = sqrt(hash_map_value(i,2)/hash_map_count(i,1));
        end
    end


end

