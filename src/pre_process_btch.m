%% batch pre-processing (pyramid construction)
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%


function btch_out = pre_process_btch(btch,S)
btch_out = zeros(size(btch,1), size(btch,2), 12, size(btch,4),'like',bthc);
for i = 1 : size(btch,4)
    pyr = laplacian_pyramid_(btch(:,:,:,i),4);
    btch_out(:,:,1:3,i) = pyr{1} .* S(1,i);
    btch_out(1:size(pyr{2},1),1:size(pyr{2},2),4:6,i) = pyr{2} * S(2,i);
    btch_out(1:size(pyr{3},1),1:size(pyr{3},2),7:9,i) = pyr{3} * S(3,i);
    btch_out(1:size(pyr{4},1),1:size(pyr{4},2),10:12,i) = pyr{4} * S(4,i);
end
btch_out = btch_out * 255;
end