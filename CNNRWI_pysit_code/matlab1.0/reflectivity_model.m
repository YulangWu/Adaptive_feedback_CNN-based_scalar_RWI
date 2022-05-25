function [ mig_reflectivity ] = reflectivity_model( vp )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [nz nx] = size(vp);
    mig_reflectivity = zeros(nz,nx);
    for i = 1 : nx
        for j = 2 : nz
            mig_reflectivity(j,i) = (vp(j,i)-vp(j-1,i))/(vp(j,i)+vp(j-1,i));
        end
    end
end

