function [ bmap,colormap ] = extract_whole_model_boundary( map, nz, nx )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    bmap = zeros(length(map),1);
    map = sign(map);
    for i = 1: length(map(1,:))
        %first assign i to each cluster
        %reverse the order just because shallow part of Marmousi is bright
        %but in common, we would like to see the dark at shallow
        bmap = bmap + i*map(:,length(map(1,:))-i+1);
    end
    colormap = bmap;
    bmap = reshape(bmap,nz,nx);
    %sign(abs()) ensures that boundary are picked as 1 not other values
    bmap = sign(abs(reflectivity_model(bmap)));
    
end

