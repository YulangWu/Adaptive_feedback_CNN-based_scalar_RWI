nx = sh_nx;
nz = sh_nz;
num_folder=num_threads;

mode = 'train'
input_directory = ['output'];
num_file = num_samples/num_folder; 
output_directory = [mode '_dataset'];
system(['mkdir ' output_directory]);

for k=1:num_folder
    for i = 1:num_file
        name = ['CNN_' mode '_dataset' num2str((k-1)*num_file+i) '.dat'];

        if exist([output_directory '/' name], 'file') ~= 0 
            disp([num2str(k) ' EXISTS===' name])
            continue;
        else
            disp([num2str(k) ' ' name])
            disp(name)
        end

        v = dlmread([input_directory num2str(k) '/' 'rtm' num2str(i) '.dat']);
        v1 = v(1:nz*nx);
        v1 = reshape(v1,nz,nx);
        v1 = preprocess_rtm(v1);
        v1 = reshape(v1,nz*nx,1);
        v(1:nz*nx) = v1;

        disp(name)
        disp(size(v)) 
        fid=fopen([output_directory '/' name],'wt');
        fprintf(fid,'%20.8f',v);
        fclose(fid);

    end
end
