function RMS = RMS(realu,estimu)
    [nz nx]=size(realu);
    RMS=((estimu-realu).^2./realu.^2);
    RMS =sqrt(sum(sum(RMS))/(nz*nx));

