%% reads and prepares input training patch
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function images = inRead(fileName, inSize)
if nargin == 1
    inSize = [256 256];
end
image = imresize(im2double(imread(fileName)),[inSize(1) inSize(2)]);

load('augInd.mat');

if aug_ind == 1
    image = flip(image,2);
end

pyr = laplacian_pyramid_(image,4);

images = zeros(size(image,1),size(image,2),size(image,3)*4);

images(:,:,1:3) = pyr{1};
images(1:inSize(1)/2,1:inSize(2)/2,4:6) = pyr{2};
images(1:inSize(1)/4,1:inSize(2)/4,7:9) = pyr{3};
images(1:inSize(1)/8,1:inSize(2)/8,10:12) = pyr{4};
images = images * 255;
end