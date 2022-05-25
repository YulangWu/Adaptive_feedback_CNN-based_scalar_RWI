function smooth_model = smooth_filter(model,filter,num)
    smooth_model = model;
    for i=1:num
        smooth_model = imfilter(smooth_model,filter,'replicate','same');
    end
    
end