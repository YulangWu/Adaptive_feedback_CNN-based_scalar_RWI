function [ puzzle_map ] = get_puzzle_layer( map, r, c, nz, nx, min_c, max_c, num)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    [totaln num_layers] = size(map);
    puzzle_map = zeros(nz*r,nx*c);
    order = randperm(r*c);
    index = 1;
    for i = 1:r
        for j = 1:c
            if sum(map(:,order(index))) ~= 0 %avoid show null space
                puzzle_map(1+(i-1)*nz:i*nz,1+(j-1)*nx:j*nx) = reshape(map(:,order(index)),nz,nx);
                index = index + 1;
            end
        end
    end
    
    figure(num)
    if min_c == 1 && max_c == 1
        imagesc(sign(puzzle_map));caxis([min_c max_c]);colormap('gray');
    else
        imagesc(puzzle_map);caxis([min_c max_c]);
    end
    set(gca, 'XTick', [])            
    set(gca,'XTickLabel',{''}) 
    set(gca, 'YTick', [])          
    set(gca,'YTickLabel',{''})  
    for i = 1:r-1
        line([1 c*nx], [i*nz i*nz],'Color','k','LineWidth',2);hold on;
    end
    for j = 1:c-1
        line([j*nx j*nx], [1 r*nz],'Color','k','LineWidth',2);hold on;
    end
        
    hold off;
end

