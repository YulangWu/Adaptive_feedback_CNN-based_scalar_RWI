function R2 = R2(realu,estimu)
    [nz nx]=size(realu);
    
    crosscorr_xy = 0.0;
    sum_x = 0.0;
    sum_y = 0.0;
    sum_xy = 0.0;
    
    avg_x = sum(sum(realu))/(nz*nx);
    avg_y = sum(sum(estimu))/(nz*nx);
    
    for i = 1:nz
        for j=1:nx
            X = real(realu(i,j)) - avg_x;
            sum_x = sum_x + (X * X);

            Y = real(estimu(i,j)) - avg_y;
            sum_y = sum_y + (Y * Y);

            crosscorr_xy = crosscorr_xy + (X * Y);
        end
    end
    R1  = crosscorr_xy / sqrt(sum_x * sum_y);  %correlation coefficient
    R2 = R1 * R1;     % coefficient of determination