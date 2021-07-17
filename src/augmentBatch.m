%% geometric augmentation 
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function batch = augmentBatch(batch)

for i = 1 : size(batch,4)
    opt = randi(3);
    switch opt
        case 1
            batch(:,:,:,i) = flip(batch(:,:,:,i),1);
        case 2
            batch(:,:,:,i) = flip(batch(:,:,:,i),2);
        case 3
            continue;
        otherwise
            continue;
    end
end
end

