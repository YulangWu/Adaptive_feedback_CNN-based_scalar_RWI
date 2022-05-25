function [ map ] = combine_kmeans_layers( kmap)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    map = zeros(length(kmap),1);
    for i = 1: length(kmap(1,:))
        map = map + kmap(:,i);
    end
end

