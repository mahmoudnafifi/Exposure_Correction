%% reads and prepares ground truth patch
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function GTimages = gtRead(fileName, In_dir, GT_dir, inSize)

In_dir = strrep(In_dir,'..','');
GT_dir = strrep(GT_dir,'..','');

fileName = strrep(fileName, In_dir, GT_dir);

parts = strsplit(fileName,'_');

fileName = [fileName((1:end-length(parts{end})-1)),'.jpg'];

GTimage = imresize(im2double(imread(fileName)),[inSize(1) inSize(2)]);

load('augInd.mat');

if aug_ind == 1
    GTimage = flip(GTimage,2);
end

GTimages = zeros(size(GTimage,1),size(GTimage,2) ,size(GTimage,3) * 4);

pyr = gaussian_pyramid_(GTimage,4);

filter = pyramid_filter_;

GTimages(:,:,1:3) = pyr{1};

for i = 2: length(pyr)
    % upsample, and add to current level
    odd = 2*size(pyr{i}) - size(pyr{i-1});
    GTimages(1:inSize(1)/(2^(i-2)),1:inSize(2)/(2^(i-2)),...
        1+((i-1)*3):3 + ((i-1)*3))= upsample_(pyr{i},odd,filter) *2^(i-2);
end

GTimages = GTimages * 255;
end