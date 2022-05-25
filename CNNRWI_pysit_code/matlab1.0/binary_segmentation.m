function [ vp_seg ] = binary_segmentation( vp,k )
%BINARY_SEGMENTATION 
%input data (vp) should be a vector not a matrix
    vp_seg = zeros(length(vp),1);
    % extract the initial velocity at particular clusted group
    for i = 1:length(vp)
        if vp(i) ~= k
            vp_seg(i) = 0;
        else
            vp_seg(i) = 1;
        end
    end

end

