function [ map ] = get_seperated_two_models( vp_mig, kmean_num, null_num, nz, nx, num_label, max_num_group )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %0. Get trivial (original) binary segementation (not perfect!!!)
    kmean_num_set_sign = sign(binary_extraction(vp_mig,kmean_num,null_num));
    
    sum1 = sum(kmean_num_set_sign(:,1));
    sum2 = sum(kmean_num_set_sign(:,2));
    
    if sum1 ~= 0 && sum2 ~= 0 % vp_mig can be divided by two groups
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %1. assign labels to different group
        map1 = reshape(kmean_num_set_sign(:,1),nz,nx);
        map2 = reshape(kmean_num_set_sign(:,2),nz,nx);
        % get_area_recur(map,1,1) %This is out of memory!!!
        for label = 1:floor(num_label/2)
            map1 = get_area_iteration(map1,label);
        end
        for label = 1+floor(num_label/2):num_label
            map2 = get_area_iteration(map2,label);
        end
        map = map1+map2;
        map = reshape(map,nz*nx,1);

        %2. find the first two maximum groups
        max_pair = zeros(max_num_group,2);%each row contains frequency and label
        unique_map = unique(map);

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


        %3. assign the first two labels to other groups
        map = reshape(map,nz,nx);
        iteration = 1000000;
        for t = 1 : iteration
            %propagate to get clustered group
            temp_map = map; %used for checking update status at the end of loop
            for i = 2:nz-1
                for j = 2 : nx-1
                    if map(i,j) ~= 0
                        if map(i,j) ~= max_pair(1,2) && map(i,j) ~= max_pair(2,2)
                            if map(i-1,j) == max_pair(1,2) || map(i-1,j) == max_pair(2,2)
                                map(i,j) = map(i-1,j);
                            elseif map(i+1,j) == max_pair(1,2) || map(i+1,j) == max_pair(2,2)
                                map(i,j) = map(i+1,j);
                            elseif map(i,j-1) == max_pair(1,2) || map(i,j-1) == max_pair(2,2)
                                map(i,j) = map(i,j-1);
                            elseif map(i,j+1) == max_pair(1,2) || map(i,j+1) == max_pair(2,2)
                                map(i,j) = map(i,j+1);
                            else
                                continue;
                            end
                        end
                    end
                end
            end

            if sum(map - temp_map) == 0
                if t ~= 1
                    disp(['Early stop at ' num2str(t) ' iteration'])
                end
                break;
            end
        end
        %pad boundary
        map(:,1) = map(:,2);
        map(:,nx) = map(:,nx-1);
        map(1,:) = map(2,:);
        map(nz,:) = map(nz-1,:);
        map = reshape(map,nz*nx,1);
    
        map1 = sign(abs(map - max_pair(1,2))).*vp_mig;
        map2 = sign(abs(map - max_pair(2,2))).*vp_mig;

        map = zeros(nz*nx,2);
        map(:,1) = map1;
        map(:,2) = map2;
    elseif sum1 == 0 % sub-figure 1 contains no features!
        map = kmean_num_set_sign(:,2).*vp_mig;
    elseif sum2 == 0 % sub-figure 2 contains no features!
        map = kmean_num_set_sign(:,1).*vp_mig;
    else
        disp(['No features are input to k-mean clustering algorithm'])
        map = 0;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     figure(2)
%     subplot(3,1,1)
%     show2d(map1)
%     subplot(3,1,2)
%     show2d(map2)
%     subplot(3,1,3)
%     show2d(whole_model)

end

