function [ max_pair ] = find_max_area( map, max_num_group)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
        [nz nx] = size(map);
        map = reshape(map,nz*nx,1);
        unique_map = unique(map);
        max_pair = zeros(max_num_group,2);%[max_num label]
        % if map cannot be divided by 2 clusters, directly return this map!
    
        hash_map = zeros(max(unique_map),1);
        for i = 1:nz*nx
            if map(i) ~= 0 % exclude the blank area
                hash_map(map(i),1) = hash_map(map(i),1)  + 1;
            end
        end

        for i = 1:max_num_group
            [max_num label] = max(hash_map);
            max_pair(i,:) = [max_num label];
            hash_map(label) = 0;
        end

end

