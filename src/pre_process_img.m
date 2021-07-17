%% image pre-processing (pyramid construction)
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%


function images = pre_process_img(image,L,S)
if nargin == 2
    S = [1 1 1 1];
end
image = im2double(image);
if L == 1
    images = image*255;
else
pyr = laplacian_pyramid_(image,L);
images = zeros(size(image,1),size(image,2),size(image,3)*L);
for i = 1 : L
    images(1:size(pyr{i},1),1:size(pyr{i},2),1+3*(i-1):3+3*(i-1)) = pyr{i} * S(i);
end
images = images * 255;
end