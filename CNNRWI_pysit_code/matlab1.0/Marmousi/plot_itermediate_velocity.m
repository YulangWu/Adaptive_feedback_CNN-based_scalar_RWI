clear all
close all
clc
format short
%------------------ colorbar setting----------------------------
Ncolor=64;
lenwhite=0;
indexcolor=zeros(Ncolor*3/2-lenwhite/2,1);
for i=1:Ncolor*1.5-lenwhite/2
    indexcolor(i)=i/(Ncolor*1.5-lenwhite/2);
end
mycolor=zeros(Ncolor*3,3);
mycolor(1:Ncolor*2,1)=1;
mycolor(1+Ncolor:Ncolor*3,3)=1;
mycolor(Ncolor*1.5-lenwhite/2:Ncolor*1.5+lenwhite/2,2)=1;
mycolor(1:Ncolor*1.5-lenwhite/2,2)=indexcolor;
mycolor(1:Ncolor*1.5-lenwhite/2,3)=indexcolor;
mycolor(1+Ncolor*1.5+lenwhite/2:Ncolor*3,1)=flipud(indexcolor);
mycolor(1+Ncolor*1.5+lenwhite/2:Ncolor*3,2)=flipud(indexcolor);
mycolor=flipud(mycolor);
cvalue = 0.001;
nz = 256;
nx = 256;
dh = 12.5;
iter =1;
offset = [64 128 192];
vp_true = dlmread(['0th_true_' 'vp' '.dat']);
vp_true = reshape(vp_true,nz,nx);

vp_init = dlmread(['0th_mig_' 'vp' '.dat']);
vp_init = reshape(vp_init,nz,nx);

vp_err(1)=RMS(vp_true,vp_init);
for i = 1:iter
    
    figure(1)
    subplot(3,3,1)
    imagesc(vp_true);caxis([1.5 4.5]);axis equal;xlim([1 nx]);ylim([1 nz]);colormap('jet');
    title(['RMS = ' num2str(RMS(vp_true,vp_true))])

    
    subplot(3,3,4)
    imagesc(vp_init);caxis([1.5 4.5]);axis equal;xlim([1 nx]);ylim([1 nz]);colormap('jet');
    title(['RMS = ' num2str(RMS(vp_true,vp_init))])

    
    vp1 = dlmread([num2str(i) 'th_true_' 'vp' '.dat']);
    vp1 = reshape(vp1,nz,nx);

    vp_err(i+1)=RMS(vp_true,vp1);


    subplot(3,3,7)
    imagesc(vp1);caxis([1.5 4.5]);axis equal;xlim([1 nx]);ylim([1 nz]);colormap('jet');
    title(['RMS = ' num2str(RMS(vp_true,vp1))])


    figure(2)
    for k = 1:3
        subplot(3,3,k)
        depth=(0:1:nz-1)*dh; % The depth (m) at each vertical grid lines
        x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
        plot(vp_true(:,x_position),depth,'k','LineWidth',1.5);hold on;
        plot(vp_init(:,x_position),depth,'b','LineWidth',1.5);hold on;
        plot(vp1(:,x_position),depth,'r','LineWidth',1.5);hold off;
        set(gca,'YDir','reverse')
        xlabel('Velocity (km/s)')
        ylabel('Depth (km)')
%         set(gca,'ytick',[1 nz/4 nz/2 nz/4*3 nz]*dh) 
%         set(gca,'xtick',1.5:1.0:4.7) 
%         set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
%         set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
%         axis([1.2 5.0 1*dh (nz)*dh]) 
        set(gca,'XAxisLocation','top');
        text(0.209139784946237, -0.6682,'a)')
        title(i)
        
    end
drawnow;
pause(0.2)
end
figure(3)
subplot(3,1,1);plot(vp_err);
