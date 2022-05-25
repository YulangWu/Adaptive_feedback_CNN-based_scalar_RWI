function [ vp_cluster ] = kmeans_model( vp_mig,kmean_num,option )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% K-means will combine similar values to one cluster so the shallow parts
% in Marmousi2 will belong to one group, so all details are missing!
    [nz nx] = size(vp_mig);
    vp_mig = reshape(vp_mig,nz*nx,1);
    
    flag = 0;
    count = 0;
    while flag == 0
        try
            vp_kmean = kmeans(vp_mig,kmean_num);
        catch ME
            flag = 0;
            count = count + 1;
            if count > 100  && mod(count,10) == 0
                disp(['repeat at ' num2str(count) ' times']);
            end
            continue;
        end
        flag = 1;
    end

    
    vp_mig = reshape(vp_mig,nz,nx);
    vp_kmean = reshape(vp_kmean,nz,nx); %get index image
    vp_mean_min_max_list = zeros(kmean_num,3);
    vp_cluster = zeros(nz,nx);

    for i = 1:kmean_num
        mean_val = 0;mean_count = 0;
        max_val = 0;
        min_val = 10;

        for ix = 1:nx
            for iz = 1:nz
                if vp_kmean(iz,ix) == i %find the grid of i-th cluster 
                    mean_val = mean_val + vp_mig(iz,ix);
                    mean_count = mean_count + 1;

                    if vp_mig(iz,ix) > max_val
                        max_val = vp_mig(iz,ix);
                    end

                    if vp_mig(iz,ix) < min_val
                        min_val = vp_mig(iz,ix);
                    end
                end
            end
        end
        vp_mean_min_max_list(i,1) = mean_val/mean_count;
        vp_mean_min_max_list(i,2) = min_val;
        vp_mean_min_max_list(i,3) = max_val;
    end

    for ix = 1:nx
        for iz = 1:nz
            vp_cluster(iz,ix) = vp_mean_min_max_list(vp_kmean(iz,ix),option);
        end
    end
end

