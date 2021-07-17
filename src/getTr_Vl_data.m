%% gets training data information
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function [Trdata,Vldata] = getTr_Vl_data(TrIn_dir, TrGT_dir, VlIn_dir, ...
    VlGT_dir, Tr_ImageNum, Vl_ImageNum, imgSize, ...
    minibatch)%,NumPatchsTr,NumPatchsVl)

Tr_in_images = imageDatastore(TrIn_dir);

Tr_in_images.ReadFcn = @(filename)inRead(filename,imgSize); %get training imgs

Tr_gt_images = imageDatastore(TrIn_dir); %get training imgs GT

Tr_gt_images.ReadFcn = @(filename)gtRead(filename, ...
    TrIn_dir, TrGT_dir, imgSize);

if Tr_ImageNum ~=0  && Tr_ImageNum < length(Tr_in_images.Files) %get random images instead of using the entire dataset
    inds = randperm(Tr_ImageNum);
    Tr_in_images.Files = Tr_in_images.Files(inds(1:Tr_ImageNum));
    Tr_gt_images.Files = Tr_gt_images.Files(inds(1:Tr_ImageNum));
end


Vl_in_images = imageDatastore(VlIn_dir); %get validation imgs

Vl_in_images.ReadFcn = @(filename)inRead(filename,imgSize); %get validation imgs

Vl_gt_images = imageDatastore(VlIn_dir); %get validation imgs GT

Vl_gt_images.ReadFcn = @(filename)gtRead(filename, ...
    VlIn_dir, VlGT_dir, imgSize);

if Vl_ImageNum ~=0  && Vl_ImageNum < length(Vl_in_images.Files) %get random images instead of using the entire dataset
    inds = randperm(Vl_ImageNum);
    Vl_in_images.Files = Vl_in_images.Files(inds(1:Vl_ImageNum));
    Vl_gt_images.Files = Vl_gt_images.Files(inds(1:Vl_ImageNum));
end

Trdata = randomPatchExtractionDatastore(Tr_in_images,Tr_gt_images, ...
    imgSize,'PatchesPerImage', 1);
Trdata.MiniBatchSize = minibatch;


Vldata = randomPatchExtractionDatastore(Vl_in_images,Vl_gt_images, ...
    imgSize,'PatchesPerImage', 1);
Vldata.MiniBatchSize = minibatch;