%% create our main network 
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function [dlnetDiscriminator] = createDiscriminator()

layersDiscriminator = imageInputLayer([256 256 3],'Normalization',...
    'none','Name','in-D');

numFilters = 8;

padding = 1;
stride = 2;
filterSize = 4;
for i = 1 : 7
    if i == 1
        layersDiscriminator = ...
            addBlockD(layersDiscriminator,filterSize, numFilters, ...
            padding,stride,0,i);
    elseif i == 7
        layersDiscriminator = ...
            addBlockD(layersDiscriminator,filterSize, ...
            min(numFilters * 2.^(i-1),256), padding,stride,1,i);
    else
        layersDiscriminator = ...
            addBlockD(layersDiscriminator,filterSize, ...
            min(numFilters * 2.^(i-1),128), padding,stride,0,i);
    end
end


lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
end


function D = addBlockD(D,filterSize, numFilters,padding,stride,F,i, scale)
if nargin == 7
    scale = 0.01;
end
if i == 2
    bn = 1;
else
    bn = 0;
end
if F == 0
    if bn == 1
        D = [D ...
            convolution2dLayer(filterSize,numFilters,'Stride',stride,...
            'Padding',padding,'Name',sprintf('conv%d-D',i)) ...
            batchNormalizationLayer('Name',sprintf('bn%d-D',i)) ...
            leakyReluLayer(scale,'Name',sprintf('lrelu%d-D',i))];
    else
        D = [D ...
            convolution2dLayer(filterSize,numFilters,'Stride',stride,...
            'Padding',padding,'Name',sprintf('conv%d-D',i)) ...
            leakyReluLayer(scale,'Name',sprintf('lrelu%d-D',i))];
    end
else
    D = [D ...
        convolution2dLayer(filterSize,numFilters,'Stride',stride,...
        'Padding',padding,'Name',sprintf('conv%d-D',i)),...
        leakyReluLayer(scale,'Name',sprintf('lrelu%d-D',i)),...
        convolution2dLayer(2,1,'Stride',2,...
        'Padding',0,'Name',sprintf('final-conv-D',i))];
end

end
