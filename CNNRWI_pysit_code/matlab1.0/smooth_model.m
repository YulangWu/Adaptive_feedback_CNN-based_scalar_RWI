function [ return_model ] = smooth_model( input_model, iters, size )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    vp_smooth = input_model;
    num_smooth_iteration = iters;
    for i = 1:num_smooth_iteration
        vp_smooth = smooth_filter(vp_smooth,fspecial('gaussian'),size);
    end
    return_model = vp_smooth;

end

