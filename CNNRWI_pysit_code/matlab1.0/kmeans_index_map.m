function [ vp_kmean ] = kmeans_index_map( vp_mig,kmean_num,max_iteration)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% K-means will combine similar values to one cluster so the shallow parts
% in Marmousi2 will belong to one group, so all details are missing!
% The input vp_mig should be a vector not matrix!
    flag = 0;
    count = 0;
    while flag == 0
        try
            % DO NOT USE SINGLETON! It will decrease the number of
            % features!
            vp_kmean = kmeans(vp_mig,kmean_num); %,'EmptyAction','singleton');%,'Start',matrix);
        catch ME
            flag = 0;
            count = count + 1;
            if mod(count,500) == 0 
                disp(['repeat at ' num2str(count) ' times']);
            end
            if count >= max_iteration
                %if you choose to decrease the cluster number, use it below
                count = 0;
                kmean_num = kmean_num - 1;
                disp(['change kmean_num from ' num2str(kmean_num+1) ' to ' num2str(kmean_num)]);
            end
            
            continue;
        end
        flag = 1;
    end 
end

