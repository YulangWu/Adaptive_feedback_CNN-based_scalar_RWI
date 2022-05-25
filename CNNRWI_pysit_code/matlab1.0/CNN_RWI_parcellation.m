% =============================================================
% Parcellation method to create training models from prior model
% =============================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The parameters input from the all_in.sh
nz=sh_nz; 
nx=sh_nx;
ratio=sh_ratio;
filename =  sh_name;
input_dir = sh_input_dir;
threshold = floor(ratio*nz*nx);
num_new_model = num_samples; %number of training velocity models

% The hyper-parameters defined here
kmean_num = 2; % Binary segmentation based on built-in K-means clustering
max_num_group = 2; % Number of candiate regions for binary segmentation
tree_depth = 14;   % maximum depth of tree
std_factor = 2;    % v = v_mean + v_std*factor

% The minimum and maximum velocities in each new generated model
min_vp = 1.4;
max_vp = 5.0;
% The minimum and maximum Pearson coefficients of new models 
% with respect to prior models (vp_mig)
min_r2 = 0.5;
max_r2 = 1.0;

% Create the output directory to store training velocity models
vel_dir = 'velocity';
system(['mkdir ' vel_dir])

disp(['tree_depth = ' num2str(tree_depth)])
disp(['threshold = ' num2str(threshold)])
disp(['num_new_model = ' num2str(num_new_model)])
disp(['std_factor = ' num2str(std_factor)])
disp(['min_vp = ' num2str(min_vp)])
disp(['max_vp = ' num2str(max_vp)])
disp(['min_r2 = ' num2str(min_r2)])
disp(['max_r2 = ' num2str(max_r2)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. binary-classification of model using k-means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.1 Input the prior model
disp(filename);
vp_mig=dlmread([input_dir filename 'vp.dat'])';
vp_mig = reshape(vp_mig,nz*nx,1);

% 1.2 Define tree structure for hierarchical parcellation
tree = struct('lchild',0,'rchild',0,'value',0);

% 1.3 Model parcellation
root = tree;
root.value = vp_mig;
lchild = tree;
rchild = tree;

%1.4 Get binary segmentation for root
[lchild.value, rchild.value]= ...
 get_seperated_two_models_for_tree_structure_only...
 (root.value,kmean_num, 0, nz, nx, max_num_group);
root.lchild = lchild;
root.rchild = rchild;

%1.5 Get binary segmentation for children
root.lchild = ...
 recurr_kmean_decomposition_for_tree_structure_only...
 (root.lchild, kmean_num, 1, nz, nx, max_num_group, tree_depth, threshold);
root.rchild = ...
 recurr_kmean_decomposition_for_tree_structure_only...
 (root.rchild, kmean_num, 1, nz, nx, max_num_group, tree_depth, threshold);

%1.6 save the leaves only as model features
segmented_vp = get_model_from_tree_at_leaves(root, nz, nx, tree_depth);

%1.7 Crop feature layers by deleting null feature
num_feature_layers = 0;
for i = 1:length(segmented_vp(1,:))
    if sum(segmented_vp(:,i)) ~= 0
        num_feature_layers = num_feature_layers + 1;
    end
end
segmented_vp = segmented_vp(:,1:num_feature_layers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Create training models from model features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.1 check whether segmentation is correct 
%    (i.e., reconstruction is correct)
cmap_vp = combine_kmeans_layers_boundary(segmented_vp);
disp(['vp error=' num2str(sum(cmap_vp-vp_mig))])

if sum(cmap_vp-vp_mig) < 10 
    %2.2 get mean and std for each feature map 
    hash_map_value_vp = get_mean_std(segmented_vp);

    %2.3 assign random value drawn from each feature to itself
    r2_value = zeros(num_new_model,1);
    vp_new_set = zeros(nz*nx,num_new_model);
    
    % Only training models satisfies minimum and maximum of velocity
    % and Pearson coefficients with respect to the prior model are stored
    k = 1;
    while k <= num_new_model
        hash_map_value_vp_randn = normal_dist...
            (hash_map_value_vp(:,1),...
            hash_map_value_vp(:,2)*std_factor,...
            min_vp,max_vp);
        
        vp_new = zeros(nz*nx,1);
        for i = 1:length(segmented_vp(1,:))
                vp_new = vp_new + ...
                    sign(segmented_vp(:,i))*hash_map_value_vp_randn(i);
        end
        
        % Check Pearson coefficients
        r2_value(k,1) = R2(vp_mig,vp_new);
        if r2_value(k,1) < min_r2 || r2_value(k,1) > max_r2
            continue; %skip this generated model
        else
            fid=fopen([vel_dir '/' 'new_vpmodel' num2str(k) '.dat'],'wt');
            fprintf(fid,'%9.6f',vp_new);
            fclose(fid);
            vp_new_set(:,k) = vp_new;
            k = k + 1; 
        end
    end
end

clear all
close all
clc