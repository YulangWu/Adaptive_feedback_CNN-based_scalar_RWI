function [ root ] = recurr_kmean_decomposition_for_tree_structure_only(root, kmean_num, null_num, nz, nx, max_num_group,tree_depth, threshold)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
    % root.value contains one image
    tree = struct('lchild',0,'rchild',0,'value',0);
    disp(['depth=' num2str(tree_depth) ' total_value=' num2str(sum(sign(root.value))) ' threshold=' num2str(threshold)]);
    if tree_depth > 0 && length(root.value) ~= 1 
        if sum(sign(root.value)) >= threshold
            tree_depth = tree_depth - 1; %So, the next tree_depth will be reduced by 1
            [layer1,layer2] = get_seperated_two_models_for_tree_structure_only(root.value,kmean_num, null_num, nz, nx, max_num_group);
            if length(layer1) ~= 1 % have feature
                lchild = tree;
                lchild.value = layer1;
                lchild = recurr_kmean_decomposition_for_tree_structure_only(lchild, kmean_num, null_num, nz, nx, max_num_group, tree_depth, threshold);
                root.lchild = lchild;
            end

            if length(layer2) ~= 1 % have feature
                rchild = tree;
                rchild.value = layer2;
                rchild = recurr_kmean_decomposition_for_tree_structure_only(rchild, kmean_num, null_num, nz, nx, max_num_group, tree_depth, threshold);
                root.rchild = rchild;
            end
    %         disp('===========================================')
    %         disp(layer1);
    %         disp(layer2);
            if length(layer1) == 1 && length(layer2)  == 1
                root.lchild = tree;
                root.rchild = tree;
            end
        end
    end
    
end

