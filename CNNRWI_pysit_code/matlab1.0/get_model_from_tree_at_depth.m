function [ layers ] = get_model_from_tree( root,tree_depth,layers, threshold)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    tree = struct('lchild',0,'rchild',0,'value',0);
    index = 1;
    
    get_model_recurr(root,1)
    
    function get_model_recurr(root,depth)
        
        if depth <= tree_depth &&  isstruct(root)
            if depth == tree_depth
                disp(['Depth = ' num2str(depth)]);
                layers(:,index) = root.value;
                index = index + 1;
            else
                disp('zeros==========')
            end
            
            depth = depth + 1;
%             if sum(sign(root.value)) >= threshold
                get_model_recurr(root.lchild,depth);
                get_model_recurr(root.rchild,depth);
%             end
        end
    end

end

