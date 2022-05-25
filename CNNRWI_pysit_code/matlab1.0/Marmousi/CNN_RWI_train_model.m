nx=sh_nx;
nz=sh_nz;
iter = sh_iter;

%%% vp model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vp = dlmread([iter 'th_true_' 'vp'  '.dat']);
vp = reshape(vp,nz,nx);


figure(1);imagesc(vp);
title('True model');
%output correct model:
vp = reshape(vp,1,nz*nx);

fid=fopen([iter 'th_true_' 'vp'  '.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);

vp = reshape(vp,nz,nx);
vp_smooth = vp;
num_smooth_iteration = 1;
for i = 1:num_smooth_iteration
    vp_smooth = imfilter(vp_smooth, fspecial('gaussian',3),'replicate','same');
end


vp_smooth=reshape(vp_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_' 'vp'  '.dat'],'wt');
fprintf(fid,'%17.8f',vp_smooth);
fclose(fid);