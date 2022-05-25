function [  ] = show_each_figure( map, nz, nx, min_c, max_c,depth)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    [totaln num_layers] = size(map);
    for i = 1:num_layers
        if sum(map(:,i)) ~= 0 %avoid show null space
            figure(i)
            show2d(map(:,i), nz, nx);
            caxis([min_c max_c]);axis equal;xlim([1 nx]);ylim([1 nz]);
            print('-depsc2','-r600',[num2str(depth) '_' num2str(i)])
        end
    end
end

