function [ map1, map2 ] = get_seperated_two_models_for_tree_structure_only( vp_mig, kmean_num, null_num, nz, nx, max_num_group )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %%%% debug: Notice that if num_label is not larger enough,
    %%%% some of segmentation will not be correctly labeled but remains 1!
    %%%% Big bug founded at 8:01 am on 02/26/2021
    %%%% So, consider a model has nz*nx points, which is the maximum
    %%%% labels we could assign to the model
    num_label = nz*nx;
    
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
        for label = 1+1:1+floor(num_label/2)
            temp_map = map1;
            map1 = get_area_iteration(map1,label);
            if sum(map1 - temp_map) == 0
                disp(['Early stop at ' num2str(label) ' label'])
                break;
            end
        end
        for label = 1+1+floor(num_label/2):1+num_label
            temp_map = map2;
            map2 = get_area_iteration(map2,label);
            if sum(map2 - temp_map) == 0
                disp(['Early stop at ' num2str(label) ' label'])
                break;
            end
        end
        
       
        %2. find the largest region in each of two group
        %If find the largest groups in the combined region there will be
        %error!!!!!!! DEBUG RESULT
        max_pair(1,:) = find_max_area( map1, 1);
        max_pair(2,:) = find_max_area( map2, 1);
        % if map cannot be divided by 2 clusters, directly return this map!
        map = map1+map2;



        %3. assign the first two labels to other groups
        map = reshape(map,nz,nx);
        iteration = 1000000;
        for t = 1 : iteration
            %propagate to get clustered group
            temp_map = map; %used for checking update status at the end of loop
            for i = 1:nz
                for j = 1 : nx
                    if map(i,j) ~= 0
                        if map(i,j) ~= max_pair(1,2) && map(i,j) ~= max_pair(2,2)
                            if i>1
                                if map(i-1,j) == max_pair(1,2) || map(i-1,j) == max_pair(2,2)
                                    map(i,j) = map(i-1,j);
                                end
                            end
                            if i<nz
                                if map(i+1,j) == max_pair(1,2) || map(i+1,j) == max_pair(2,2)
                                    map(i,j) = map(i+1,j);
                                end
                            end
                            if j>1
                                if map(i,j-1) == max_pair(1,2) || map(i,j-1) == max_pair(2,2)
                                    map(i,j) = map(i,j-1);
                                end
                            end
                            if j<nx
                                if map(i,j+1) == max_pair(1,2) || map(i,j+1) == max_pair(2,2)
                                    map(i,j) = map(i,j+1);
                                end
                            else
                                continue;
                            end
                        end
                    end
                end
            end

            if sum(map - temp_map) == 0
                if t > iteration/1000 && mod(t,100) == 0
                    disp(['Early stop at ' num2str(t) ' iteration'])
                end
                break;
            end
        end
%         %pad boundary
%         map(:,1) = map(:,2);
%         map(:,nx) = map(:,nx-1);
%         map(1,:) = map(2,:);
%         map(nz,:) = map(nz-1,:);
        map = reshape(map,nz*nx,1);
    
        map1 = sign(abs(map - max_pair(1,2))).*vp_mig;
        map2 = sign(abs(map - max_pair(2,2))).*vp_mig;
    elseif sum1 == 0 % sub-figure 1 contains no features!
        map1 = 0;
        map2 = kmean_num_set_sign(:,2).*vp_mig;
    elseif sum2 == 0 % sub-figure 2 contains no features!
        map1 = kmean_num_set_sign(:,1).*vp_mig;
        map2 = 0;
    else
        disp(['No features are input to k-mean clustering algorithm'])
        map1 = 0;
        map2 = 0;
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

