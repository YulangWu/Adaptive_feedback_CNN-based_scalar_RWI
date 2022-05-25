function smooth_model = smooth(model, num)
    [nz nx] = size(model);
    filter=ones(3,3);
    smooth_model=model;
    filter(1,1)=0;
    filter(3,1)=0;
    filter(1,3)=0;
    filter(3,3)=0;
    for i=1:num
    model= xcorr2(smooth_model,filter);
    smooth_model=model(2:nz+1,2:nx+1);
    smooth_model(1,:)  = smooth_model(2,:);
    smooth_model(nz,:) = smooth_model(nz-1,:);
    smooth_model(:,1)  = smooth_model(:,2);
    smooth_model(:,nx) = smooth_model(:,nx-1);
    smooth_model=smooth_model/5;
    end
    
end