function [ out_layer ] = extract_individual_kmeans_layers_boundary(in_layer, nz, nx)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [totaln,num_maps] = size(in_layer);
    out_layer = zeros(totaln,num_maps);
    for i = 1:num_maps
        input_map = in_layer(:,i);
        input_map = reshape(input_map,nz,nx);
        out_map = ...
            boundary_extraction_morphology( input_map, nz, nx );
        out_map = reshape(out_map,nz*nx,1);
        out_layer(:,i) = out_map;   
    end
end

