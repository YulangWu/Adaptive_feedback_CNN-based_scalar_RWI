function [ layers ] = get_model_from_tree_at_leaves(root, nz, nx, max_depth)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    tree = struct('lchild',0,'rchild',0,'value',0);
    index = 1;
    layers = zeros(nz*nx,2^(max_depth));%num of leaves at certain depth
    get_model_recurr_leave(root);
    
    function get_model_recurr_leave(root)
        if ~isstruct(root.lchild) && ~isstruct(root.rchild)
%             disp('find leave')
            layers(:,index) = root.value;
            index = index + 1;
            
        end
        if isstruct(root.lchild)
%             disp('find lchild')
            get_model_recurr_leave(root.lchild);
        end
        if isstruct(root.rchild)
%             disp('find rchild')
            get_model_recurr_leave(root.rchild);
        end
    end

end

