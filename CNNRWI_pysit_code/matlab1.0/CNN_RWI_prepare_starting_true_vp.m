input_directory = sh_input_dir;
nz = sh_nz;
nx = sh_nx;
pysit_dir=sh_pysit_dir;
output_directory = ['../' pysit_dir 'given_models'];
system(['mkdir ' output_directory num2str(0)]);

type = 'vp'

true_model = dlmread([input_directory '0th_true_' type '.dat']);
true_model = reshape(true_model,1,nz*nx);

fid=fopen([output_directory num2str(0) '/' 'true' num2str(0) type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',true_model);
fclose(fid);

smooth_model = dlmread([input_directory '0th_mig_' type '.dat']);
smooth_model = reshape(smooth_model,1,nz*nx);

fid=fopen([output_directory num2str(0) '/' 'mig' num2str(0) type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',smooth_model);
fclose(fid);
