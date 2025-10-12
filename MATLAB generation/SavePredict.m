clc;clear;close all;
%% merge
save_path='G:\datasets\microtubules';
bsize=1024;
interval=1;
flage=1;
flage_est=1;
for i=[2:9,11,12]
    raw=ReadTiff(['cell',num2str(i),'.tif']);
    re=ReadTiff(['re-cell',num2str(i),'.tif']);
    [x,y,t]=size(re);
    xn=floor((x-(bsize-interval))/interval);
    yn=floor((y-(bsize-interval))/interval);
    %     xn=round(x./bsize);
    %     yn=round(y./bsize);
    xl=x-xn*bsize;
    yl=y-yn*bsize;
    ts=size(raw,3)/9;
    ts=floor(ts*0.9)*9;
    ts=9;
    reflage=ts/9;
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
                if sum(re2save(:))>120
                    re2save=re2save./max(re2save(:));
                    raw2save=raw2save./max(raw2save(:));
                    img2save=[re2save,raw2save];
                    imwrite(double(img2save),[save_path,'\predict\',num2str(flage_est),'.tif'])
                    flage_est=flage_est+1;
                end
            end
        end
    end
    
end
% SaveTest;