nz = sh_nz;
nx = sh_nx;

input_vp_filename = sh_input_vp_filename;
output_vp_filename = sh_output_vp_filename;

vp_data = dlmread(input_vp_filename);

CNN_output_vp= vp_data(1+nz*nx*3:nz*nx*4)/1000;
CNN_output_vp = reshape(CNN_output_vp,nz,nx);

vp = reshape(CNN_output_vp,1,nz*nx);
fid=fopen([output_vp_filename '.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);
