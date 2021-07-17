%% main code to extract random patches with different dimensions
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

% please, adjust directory accordingly 

dataset_dir = 'exposure_dataset';

if exist(dataset_dir, 'dir') == 0
    dataset_dir = fullfile('..', dataset_dir);
end

if exist(dataset_dir, 'dir') == 0
    error('Dataset directory does not exist');
end

In_Tr_datasetDir = fullfile(dataset_dir,'training','INPUT_IMAGES');

GT_Tr_datasetDir = fullfile(dataset_dir,'training','GT_IMAGES');

In_Vl_datasetDir = fullfile(dataset_dir,'validation','INPUT_IMAGES');

GT_Vl_datasetDir = fullfile(dataset_dir,'validation','GT_IMAGES');

for i = 7:9
    ps = 2^i;
    patchSize = [ps, ps ,3];
    tr_num_pch_per_image = 2.^(10 - i);
    vl_num_pch_per_image = 2.^(10 - i)/2;
    
    Out_Tr_datasetDir = [In_Tr_datasetDir sprintf('_P_%d',ps)];
    
    GT_Out_Tr_datasetDir = [GT_Tr_datasetDir sprintf('_P_%d',ps)];
    
    Out_Vl_datasetDir = [In_Vl_datasetDir sprintf('_P_%d',ps)];
    
    GT_Out_Vl_datasetDir = [GT_Vl_datasetDir sprintf('_P_%d',ps)];
    
    fprintf('processing training images (%d,%d)...\n',ps,ps);
    
    randPatchExtraction(In_Tr_datasetDir,GT_Tr_datasetDir,...
        Out_Tr_datasetDir, GT_Out_Tr_datasetDir, patchSize, tr_num_pch_per_image);
    
    fprintf('processing validation images (%d,%d)...\n',ps,ps);
    
    randPatchExtraction(In_Vl_datasetDir,GT_Vl_datasetDir,...
        Out_Vl_datasetDir, GT_Out_Vl_datasetDir, patchSize, vl_num_pch_per_image);
    
end
