function [ v1 ] = preprocess_rtm( v1 )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [nz nx] = size(v1);
    v1_sm = smooth_filter(v1,fspecial('gaussian'),2);
    v1 = v1 - v1_sm;
    %boost the amplitude in the lower part
    for i = 1:nz
        v1(i,:) = v1(i,:)*i^2;
    end

end

