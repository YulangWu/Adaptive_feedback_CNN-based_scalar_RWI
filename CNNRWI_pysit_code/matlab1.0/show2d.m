function [  ] = show2d( data, nz, nx)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    data =reshape(data,nz,nx);
    imagesc(data)
    set(gca, 'XTick', [])          
    set(gca,'XTickLabel',{''}) 
    set(gca, 'YTick', [])          
    set(gca,'YTickLabel',{''}) 
end

