function [ map ] = get_area_iteration( map, label)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % This function naturally exclude the blank area since
    % it only judge whether the value is 1 or -1 (ignore 0 for blank area);
    iteration = 1000000;
    [nz nx]= size(map);
    
    flag = 0;
    for i = 1:nz
        if flag == 1 %means first 1 is founded and marked as -1
            break;
        end
        
        for j = 1 : nx
            if map(i,j) == 1
                map(i,j) = -1;
                flag = 1;
                break;
            end
        end
    end

    map = reshape(map,nz,nx);
    % get_area_recur(map,1,1) %This is out of memory!!!

    count = 0; % count the number of points
    for t = 1:iteration
        %propagate to get clustered group
        temp_map = map; %used for checking update status at the end of loop
        for i = 1:nz
            for j = 1:nx
                if map(i,j) == -1
                    if i + 1 <= nz
                        if map(i+1,j) == 1
                            map(i+1,j) = -1;
                            count = count + 1;
                        end
                    end

                    if i - 1 >= 1
                        if map(i-1,j) == 1
                            map(i-1,j) = -1;
                            count = count + 1;
                        end
                    end

                    if j + 1 <= nx
                        if map(i,j+1) == 1
                            map(i,j+1) = -1;
                            count = count + 1;
                        end
                    end

                    if j - 1 >= 1
                        if map(i,j-1) == 1
                            map(i,j-1) = -1;
                            count = count + 1;
                        end
                    end
                end
            end
        end
        if sum(map - temp_map) == 0
%             if t ~= 1
            if t > iteration/1000 && mod(t,100) == 0
                disp(['Early stop at ' num2str(t) ' iteration'])
            end
            break;
        end
    end
    
    % get color
    for i = 1:nz
        for j = 1:nx
            if map(i,j) == -1
                map(i,j) = label; %-count; %is used to get number of points
            end
        end
    end
end

