clc;clear;close all;
%% merge
save_path='G:\datasets\microtubules';
bsize=256;
interval=128;
flage=1;
flage_est=1;
backg1 = 2;
backg2 = 2;
for i=[1,20]
    raw=ReadTiff(['G:\DL-SIM\microtubules\c',num2str(i),'.tif']);
    re=ReadTiff(['G:\DL-SIM\microtubules\re-c',num2str(i),'.tif']);
    backgrounds= background_estimation(raw./backg1,1,7,'db6',3);
    raw=raw-backgrounds;
    raw(raw<0)=0;
    backgrounds= background_estimation(re./backg2,1,7,'db6',3);
    re=re-backgrounds;
    re(re<0)=0;
    [x,y,t]=size(re);
    xn=floor((x-(bsize-interval))/interval);
    yn=floor((y-(bsize-interval))/interval);
    xl=x-xn*bsize;
    yl=y-yn*bsize;
    reflage=1;
    ts=size(raw,3)/9; 
    ts=floor(ts*1)*9;

    for j=1:9:ts
        rawmerge=raw(:,:,j)+raw(:,:,j+3)+raw(:,:,j+6);
        rawmerge=rawmerge-min(rawmerge(:));
        rawmerge=rawmerge./max(max(rawmerge));
        rawmerge2=fourierInterpolation( rawmerge, 2, 'lateral' );
        rawmerge2=rawmerge2./max(max(rawmerge2));
        res=re(:,:,reflage)./max(max(re(:,:,reflage)));
        res=res-min(res(:));
        res(res<0)=0;
        res=res./max(res(:));
        reflage=reflage+1;
        for xx=1:xn
            for yy=1:yn
                re2save=res((xx-1)*interval+1:(xx-1)*interval+bsize  ,  (yy-1)*interval+1:(yy-1)*interval+bsize );
                raw2save=rawmerge2((xx-1)*interval+1:(xx-1)*interval+bsize  ,  (yy-1)*interval+1:(yy-1)*interval+bsize );
                if sum(re2save(:))>200
                    re2save=re2save./max(re2save(:));
                    raw2save=raw2save./max(raw2save(:));
                    img2save=[re2save,raw2save];
                    imwrite(double(img2save),[save_path,'\train\',num2str(flage),'.tif'])
                    flage=flage+1;
                end
            end
        end
    end
    disp(['Number ', num2str(i), ', frames ', num2str(flage)])
    xn=round(x./bsize);
    yn=round(y./bsize);
    for j=ts+1
        rawmerge=raw(:,:,j)+raw(:,:,j+3)+raw(:,:,j+6);
        rawmerge=rawmerge./max(max(rawmerge));
        rawmerge2=fourierInterpolation( rawmerge, 2, 'lateral' );
        rawmerge2=rawmerge2./max(max(rawmerge2));
        res=re(:,:,reflage)./max(max(re(:,:,reflage)));
        %         res=res-min(res(:));
        res=res./max(res(:));
        reflage=reflage+1;
        for xx=1:xn
            for yy=1:yn
                re2save=res((xx-1)*bsize+1:(xx)*bsize,(yy-1)*bsize+1:(yy)*bsize);
                raw2save=rawmerge2((xx-1)*bsize+1:(xx)*bsize,  (yy-1)*bsize+1:(yy)*bsize);
                if sum(re2save(:))>200
                    re2save=re2save./max(re2save(:));
                    raw2save=raw2save./max(raw2save(:));
                    img2save=[re2save,raw2save];
                    imwrite(double(img2save),[save_path,'\test\',num2str(flage_est),'.tif'])
                    flage_est=flage_est+1;
                end
            end
        end
    end
    
end
% SaveTest;