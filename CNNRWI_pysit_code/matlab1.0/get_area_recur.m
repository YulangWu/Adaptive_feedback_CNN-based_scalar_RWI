function [ map ] = get_area_recur(map,start_z,start_x)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
%   map(z,x) = 1 and has been marked as -1 before going here
    [nz nx] = size(map);
    get_area(start_z,start_x);
    imagesc(map);
    
    function get_area(z,x)
        if z > 1
            if map(z-1,x) == 1
                map(z-1,x) = -1;
                get_area(z-1,x);
            end
        end

        if z < nz
            if map(z+1,x) == 1
                map(z+1,x) = -1;
                get_area(z+1,x);
            end
        end

        if x > 1
            if map(z,x-1) == 1
                map(z,x-1) = -1;
                get_area(z,x-1);
            end
        end

        if x < nx
            if map(z,x+1) == 1
                map(z,x+1) = -1;   
                get_area(z,x+1);
            end
        end
    end
    
end

