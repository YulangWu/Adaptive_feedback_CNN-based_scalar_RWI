function [ value_vector ] = normal_dist( mu_vec,sigma_vec, min_val, max_val)
%NORMAL_DIST Summary of this function goes here
%   Detailed explanation goes here
    value_vector = mu_vec + sigma_vec.*randn(length(mu_vec),1);
    
%   add constraint to the minimum and maximum value
    for i = 1:length(value_vector)
        if value_vector(i) < min_val
            value_vector(i) = min_val;
        elseif value_vector(i) > max_val
            value_vector(i) = max_val;
        end
    end
end

