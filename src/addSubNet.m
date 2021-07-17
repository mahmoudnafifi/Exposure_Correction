%% adds sub-network to the main network
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function net = addSubNet(net, imageSize, depth, chnls, convvfilter, ...
    isLastLayer, name, islastPyrLayer)


sub_net = removeLayers(...
    removeLayers(unetLayers([imageSize(1) imageSize(2) 3] ,3, 'EncoderDepth',depth, ...
    'NumFirstEncoderFilters', chnls, 'FilterSize' , convvfilter),...
    {'Softmax-Layer','Segmentation-Layer'}),'ImageInputLayer');

indices = find(contains({sub_net.Layers(:).Name},'-ReLU-'));

for i = indices 
    leakyReLU = leakyReluLayer(0.2,'Name',strrep(sub_net.Layers(i).Name,'ReLU','L-ReLU'));
    sub_net = replaceLayer(sub_net,sub_net.Layers(i).Name,leakyReLU);
end

%removing dropOut layers
sub_net = removeLayers(sub_net,'Bridge-DropOut');
sub_net = removeLayers(sub_net,sprintf('Encoder-Stage-%d-DropOut',depth));
sub_net = connectLayers(sub_net,'Bridge-L-ReLU-2','Decoder-Stage-1-UpConv');
sub_net = connectLayers(sub_net,sprintf('Encoder-Stage-%d-L-ReLU-2',depth),...
    sprintf('Encoder-Stage-%d-MaxPool',depth));

layers = sub_net.Layers;

for i = 1 : length(layers)
    layers(i).Name = [name '-' layers(i).Name];
end

net = addLayers(net, layers);

for i = 1 : depth
    net = connectLayers(net,sprintf('%s-Encoder-Stage-%d-L-ReLU-2',name,i),...
        sprintf('%s-Decoder-Stage-%d-DepthConcatenation/in2',name, depth-i+1));
end

if islastPyrLayer == 0
    addL = additionLayer(2,'Name',sprintf('%s-reconstructLayer',name));
    net = addLayers(net, addL);
        net = connectLayers(net,[name '-Final-ConvolutionLayer'],...
        sprintf('%s-reconstructLayer/in1',name));
end
if isLastLayer == 0
    upsamplingLayer = transposedConv2dLayer([2 2],3,'Stride',[2 2],...
        'Name',[name '-upsampling']);
    net = addLayers(net,  upsamplingLayer);

    if islastPyrLayer == 1
        net = connectLayers(net,[name ...
            '-Final-ConvolutionLayer'],upsamplingLayer.Name);
    else
        net = connectLayers(net,sprintf('%s-reconstructLayer',name),...
            upsamplingLayer.Name);
    end
end

end

