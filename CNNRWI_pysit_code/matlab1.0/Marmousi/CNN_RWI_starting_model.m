nx=sh_nx;
nz=sh_nz;
filename=sh_filename;
iter = num2str(0);
num_smooth_iteration = sh_num_smooth_iteration;
filter_size = sh_filter_size;
water_depth = sh_water_depth;

vp = dlmread(filename);
vp = reshape(vp,nz,nx);

figure(1);
imagesc(vp);caxis([1.5 4.5]);colormap('jet');

title('True model');
vp = reshape(vp,1,nz*nx);

fid=fopen([iter 'th_true_vp.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);

% ========================================
% create smooth vp model
% ========================================
vp = reshape(vp,nz,nx);
vp_smooth = vp;
for i = 1:num_smooth_iteration
    if i >= num_smooth_iteration-10
        vp_smooth(1:water_depth,:) = vp(1:water_depth,:);
    end
    vp_smooth = imfilter(vp_smooth, fspecial('gaussian',filter_size),'replicate','same');
end
subplot(2,3,4);imagesc(vp_smooth);caxis([1.5 4.5]);colormap('jet');

vp_smooth=reshape(vp_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_vp.dat'],'wt');
fprintf(fid,'%17.8f',vp_smooth);
fclose(fid);