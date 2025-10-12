function data=ReadTiff(filename)

warning off
info = imfinfo(filename);
num_images = numel(info);
for k=1:num_images
    t=Tiff(filename, 'r+' );
    t.setDirectory(k);
    data(:,:,k)= single(t.read());
    % data(:,:,k)=data(:,:,k)./max(max(data(:,:,k)));
    if mod(k,10000)==0
        disp([num2str(k) 'frames'])
    end
end
