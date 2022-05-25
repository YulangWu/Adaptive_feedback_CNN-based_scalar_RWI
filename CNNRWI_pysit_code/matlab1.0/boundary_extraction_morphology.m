function [ out_map ] = boundary_extraction_morphology( input_map, nz, nx )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    %see 9.5 Some basic morphological algorithms in book:
    % << Digital Image Processing (4th edition) >>
    % Author: Rafael C. Gonzalez & Richard E. Woods
    map = reshape(input_map,nz,nx);
    bmap = zeros(nz,nx);
    for i = 2:nz-1
        for j = 2:nx-1
            if (sum(sum(map(i-1:i+1,j-1:j+1))) == map(i,j)*9) %all cells are equal
                if(map(i,j)==1)
                    bmap(i,j) = 1;
                end
            end
        end
    end
    bmap(:,1) = bmap(:,2);
    bmap(:,nx) = bmap(:,nx-1);
    bmap(1,:) = bmap(2,:);
    bmap(nz,:) = bmap(nz-1,:);
    out_map = map - bmap;

end

