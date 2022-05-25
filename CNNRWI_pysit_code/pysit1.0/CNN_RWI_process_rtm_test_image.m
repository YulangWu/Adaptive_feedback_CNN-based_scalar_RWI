nx = sh_nx;
nz = sh_nz;

mode = 'real'
input_directory = ['output'];
output_directory = [mode '_dataset'];
system(['mkdir ' output_directory]);

name = ['CNN_' mode '_dataset0.dat'];

if exist([output_directory '/' name], 'file') ~= 0 
    disp([num2str(0) ' EXISTS===' name])
else
    disp([num2str(0) ' ' name])
    disp(name)
end

figure(1)
v = dlmread([input_directory num2str(0) '/' 'rtm' num2str(0) '.dat']);
v1 = v(1:nz*nx);
v1 = reshape(v1,nz,nx);
subplot(2,2,1);imagesc(v1);colormap('gray');caxis([-1 1]);
v1 = preprocess_rtm(v1);
subplot(2,2,2);imagesc(v1);colormap('gray');caxis([-100 100]);
v1 = reshape(v1,nz*nx,1);
v(1:nz*nx) = v1;

disp(name)
disp(size(v)) 
fid=fopen([output_directory '/' name],'wt');
fprintf(fid,'%20.8f',v);
fclose(fid);


