clc;clear;close all;
addpath(genpath('F'));
% mkdir (['aged_SACD_4D_intensity_reg_roll100']);
% load A;
% A = imresize(A,[1,650]);
% A = A./max(A);
% data1 = zeros(536,536,672);
% for i1 = 1:650
% i1
% for i2 = 1:100
% i2
% data = imRS(['data_avg_',num2str((i1-1)+i2)]);
% data1 = data1+data;
% % data2(:,:,:,i2) = imresize(data,[200,200],'bilinear');
% end
% data_avg_post = (data1./100).*A(i1);
% imWS(data_avg_post,['aged_SACD_4D_intensity_reg_roll100/','data_avg_',num2str(i1)]);
% data1 = double(zeros(536,536,672));
% % data_avg_post_MIP_(:,:,i1) = max(permute(data_avg_post,[3 2 1]),[],3).*A(i1);
% end
n=3;
data = tiffreadVolume('G:/MGAN-data/Actin_pred_70_2.tif');
disp(size(data));
frames = size(data, 3);
for i = 1:frames
    startIdx = i;
    endIdx = min(i + n, frames);  % 防止越界
    data(:, :, i) = mean(data(:, :, startIdx:endIdx), 3);  % 沿时间维平均
end
imWS(data,'G:/MGAN-data/Actin_pred_70_rolling2');
% imWS(data_avg_post_MIP_,['young_n+p_SACD_4D_intensity_reg_rollavg_V2/data_avgMIP_T2']);
function [name] = imRS(file)
num_images = numel(imfinfo([file, '.tif']));
for i = 1:num_images
name(:,:,i) = double(imread([file, '.tif'],i));
end
name = name./max(max(max(name)));
end
function imWS(name,file)
name = double(name);
% name = name./max(max(max(name)));
imwrite (name(:,:,1),[file,'.tif']);
for i = 2:size(name,3)
imwrite (name(:,:,i),[file,'.tif'],'WriteMode','append');
end
end
function [output2] = reg3D(input1,input2)
Target = (input1);
Source = (input2);
[D,moving_reg] = imregdemons(Source, Target,'PyramidLevels',2,'AccumulatedFieldSmoothing',1.5);
% output1 = single(gather(Target));
output2 = single((moving_reg));
end