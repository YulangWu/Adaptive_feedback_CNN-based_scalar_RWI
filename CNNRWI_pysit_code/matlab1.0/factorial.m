function [ n ] = factorial( n )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    if n == 1
        n = 1;
    else
    n = n * factorial(n-1);
    end
end

