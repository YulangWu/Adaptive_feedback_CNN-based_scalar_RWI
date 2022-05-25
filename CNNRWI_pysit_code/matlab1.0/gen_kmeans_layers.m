function [ out_layer ] = gen_kmeans_layers( in_layer, kmean_num, nz, nx, null_num, num_label, max_num_group)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [totaln,num_maps] = size(in_layer);
    out_layer = zeros(totaln,num_maps*2);
    for i = 1:num_maps
        if sum(in_layer(:,i)) ~= 0 %in case that some of the input layer map are already full of zeros
            temp_layer = get_seperated_two_models(in_layer(:,i),kmean_num, null_num, nz, nx, num_label, max_num_group );
            [totaln num_layer] = size(temp_layer);
            if num_layer == 1
                out_layer(:,1+(i-1)*2) = temp_layer;
                %out_layer(:,1+i*2) is zeros
            else %num_layer should be 2
                out_layer(:,1+(i-1)*2:i*2) = temp_layer;
            end
        end
    end
end

