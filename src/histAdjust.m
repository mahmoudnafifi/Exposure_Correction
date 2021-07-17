%% global hist adjustment for contrast enhancement 
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

%% Note: this was not being used in our paper experiments, but it can improve results in some cases
%%

function out = histAdjust(I)
I = double(I);
param=stretchlim(I);
li = param(1);
hi = param(2);
lo = 0;
ho = 1;
gamma = 1;
out = (I < li) .* lo;
out = out + (I >= li & I < hi) .* (lo + (ho - lo) .* ((I - li) / (hi - li)) .^ gamma);
out = out + (I >= hi) .* ho;
end

