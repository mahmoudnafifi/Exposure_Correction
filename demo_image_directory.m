%% demo -- processing image directory
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%
clear;
clc
close all;
addpath('bgu'); % for guided upsampling 
addpath('exFusion'); % for post-processing fusion (optional) --  this part was not included in our paper
modelName = fullfile('models','model.mat'); % model full file name
base_dir = 'example_images/'; % input image directory 
base_dir_out = 'results'; % output directory 
exts = {'.jpg','.jpeg','.png','.tif','.bmp'}; % input image extensions
vis = 0; % visualization? to show input/output figure
device = 'gpu'; % use gpu? options: 'gpu' or 'cpu'
pp = 0; % post-processing contrast adjustment -- this part was not used in our paper
fusion = 0; % post-processing fusion of input and output images -- this part was not used in our experiments
L = 4; % number of sub-networks

if exist(base_dir_out,'dir') == 0
    mkdir(base_dir_out);
end
load(modelName);


images = {};
for i = 1 : length(exts)
    temp_files = dir(fullfile(base_dir,['*' exts{i}]));
    images = [images; {temp_files(:).name}'];
end

for i = 1 : length(images)
    imageName = images{i};
    disp('---------------------------------------------');
    fprintf('processing image %s...\n',imageName);
    disp('---------------------------------------------');
    try
        I = im2double(imread(fullfile(base_dir,imageName)));
    catch
        fprintf('Cannot read image %s!\n',imageName);
        continue;
    end
    
    %% check image size
    sz = size(I);
    mxdim = max(sz);
    inSz = 512;
    if (mxdim > inSz) == 1
      S = [1.5, 1.5, 1.5, 1.05]; % scale vector -- tunable hyper-parameter
    else
      S = [1,1,1,1];
    end
    I_ = I;
    I = imresize(I,inSz/max(sz));
    pad_factor = [inSz-size(I,1) inSz-size(I,2)];
    I = padarray(I, pad_factor,'replicate','pre');
    IMAGE = pre_process_img(I,L,S);
    disp('Exposure correction...');
    tic
    if strcmpi(device,'gpu') == 1
        output = gather(extractdata(predict(net,gpuArray(dlarray(IMAGE,...
            'SSCB')))))/255;
    else
        output = extractdata(predict(net,dlarray(IMAGE,'SSCB')))/255;
    end
    fprintf('Network processing time is %d seconds.\n',toc);
    
    output = output(pad_factor(1)+1:end,pad_factor(2)+1:end,1:3);
    I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
    
    
    if (max(sz) > inSz) == 1
        disp('Upsampling...');
        tic
        output_s = double(imresize(output,[200,200]));
        I = double(imresize(I,[200,200]));
        results = computeBGU(I, rgb2luminance(I), output_s, [], ...
            I_, rgb2luminance(I_));
        output = results.result_fs;
        fprintf('Upsampling time: %f seconds.\n',toc);
    else
        output = imresize(output,[sz(1) sz(2)]);
    end
    
    
    if fusion == 1
        disp('Fusion...');
        tic
        Out = zeros(size(output,1),size(output,2),size(output,3),2);
        Out(:,:,:,1) = I_;
        Out(:,:,:,2) = output;
        output = exposure_fusion(Out,[1 1 1]);
        fprintf('Fusion time: %f seconds.\n',toc);
    end
    
    if pp == 1
        output = histAdjust(output);
    end
    if vis == 1
        figure;
        subplot(1,2,1);imshow(I_);title('Input image');
        subplot(1,2,2);imshow(output);title('Result');
        linkaxes;
    end
    
    
    
    [~,name,ext] = fileparts(imageName);
    imwrite(output,fullfile(base_dir_out,[name ext]));
    fprintf('\n\n');
end