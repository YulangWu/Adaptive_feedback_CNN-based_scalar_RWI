nx = sh_nx;
nz = sh_nz;
num_smooth_iteration = sh_num_smooth_iteration;
filter_size = sh_filter_size;
water_depth = sh_water_depth;

mode = 'train'
num_node = num_threads;
input_directory = 'velocity';
output_dir = sh_pysit_dir
output_directory = ['../' sh_pysit_dir 'given_models'];
for i = 1 : num_node
    system(['mkdir ' output_directory num2str(i)]);
end

Files_vp=dir([input_directory '/' 'new_' 'vp' 'model*.dat']); 
   
for k=1:length(Files_vp)
    FileNames_vp=Files_vp(k).name;

    len = length(FileNames_vp);
    name = ['train' FileNames_vp(4:len-4) '.dat'];
    
    
    if exist(name, 'file') ~= 0 
        continue;
    else
        disp([num2str(k) ' ' FileNames_vp])
        disp(name)
    end
    
    true_model = dlmread([input_directory '/' FileNames_vp]);
    true_model = reshape(true_model,nz,nx);
    true_model(1:water_depth,:) = true_model(1,1);
    
    smooth_model = true_model;
    for i = 1:num_smooth_iteration
        if i >= num_smooth_iteration-10
            smooth_model(1:water_depth,:) = true_model(1:water_depth,:);
        end
        smooth_model = imfilter(smooth_model, ...
            fspecial('gaussian',filter_size),'replicate','same');
    end
    smooth_model = reshape(smooth_model,1,nz*nx);

    true_model = reshape(true_model,1,nz*nx);

    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' 'true' ...
            num2str(floor(k/num_node+1))  'vp' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' 'true' ...
            num2str(floor(k/num_node))  'vp' '.dat'],'wt');
    end
    
    disp(num2str(floor(k/num_node+1)))
    fprintf(fid,'%17.8f',true_model);
    fclose(fid);

    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' 'mig' ...
            num2str(floor(k/num_node+1))  'vp' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' 'mig' ...
            num2str(floor(k/num_node))  'vp' '.dat'],'wt');
    end
    fprintf(fid,'%17.8f',smooth_model);
    fclose(fid);
end

clear all
close all
clc