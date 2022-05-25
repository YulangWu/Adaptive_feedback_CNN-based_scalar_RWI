function [  ] = show2dlayer( map, k, r, c, nz, nx, min_c, max_c)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    figure(k)
    [totaln num_layers] = size(map);
    for i = 1:num_layers
        if sum(map(:,i)) ~= 0 %avoid show null space
            subplot(r,c,i)
            show2d(map(:,i), nz, nx);
            caxis([min_c max_c]);axis equal;xlim([1 nx]);ylim([1 nz])
        end
    end
end

