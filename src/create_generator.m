%% create our main network 
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function net = create_generator(patchSize, encoderDecoderDepth, chnls, convfilter)

if nargin ==0
    patchSize = [256, 256, 12];
    encoderDecoderDepth = 3;
    chnls = 16;
    convfilter = 3;
end

inLayer = imageInputLayer(patchSize,'Name','InputLayer',...
    'Normalization','none');
extractPyrLayer_level_1 =  extractPyrLayer('level_1_extract_pyr',1);
extractPyrLayer_level_2 =  extractPyrLayer('level_2_extract_pyr',2);
extractPyrLayer_level_3 =  extractPyrLayer('level_3_extract_pyr',3);
extractPyrLayer_level_4 =  extractPyrLayer('level_4_extract_pyr',4);

add_out_L_4_in_L3 = additionLayer(2,'Name','out_L_4_in_L_3');
add_out_L_3_in_L2 = additionLayer(2,'Name','out_L_3_in_L_2');
add_out_L_2_in_L1 = additionLayer(2,'Name','out_L_2_in_L_1');

net = layerGraph(inLayer);
net = addLayers(addLayers(addLayers(addLayers(net,extractPyrLayer_level_1),...
    extractPyrLayer_level_2),extractPyrLayer_level_3),extractPyrLayer_level_4);

net = addLayers(addLayers(addLayers(net,add_out_L_4_in_L3), ...
    add_out_L_3_in_L2), add_out_L_2_in_L1);


packingLayer = packingGLayer('packingLayer');

net = addLayers(net, packingLayer);

net = addSubNet(net, patchSize, encoderDecoderDepth, chnls, ...
    convfilter, 1, 'level_1',0);
net = addSubNet(net, patchSize, encoderDecoderDepth, chnls + 8, ...
    convfilter, 0, 'level_2',0);
net = addSubNet(net, patchSize, encoderDecoderDepth, chnls + 8, ...
    convfilter, 0, 'level_3',0);
net = addSubNet(net, patchSize, encoderDecoderDepth + 1, chnls + 8, ...
    convfilter, 0, 'level_4',1);

net = connectLayers(net, 'level_1-reconstructLayer','packingLayer/in1');
net = connectLayers(net, 'level_2-upsampling','packingLayer/in2');
net = connectLayers(net, 'level_3-upsampling','packingLayer/in3');
net = connectLayers(net, 'level_4-upsampling','packingLayer/in4');

net = connectLayers(net,'InputLayer','level_1_extract_pyr');
net = connectLayers(net,'InputLayer','level_2_extract_pyr');
net = connectLayers(net,'InputLayer','level_3_extract_pyr');
net = connectLayers(net,'InputLayer','level_4_extract_pyr');

net = connectLayers(net,'level_4-upsampling','out_L_4_in_L_3/in1');
net = connectLayers(net,'level_3_extract_pyr','out_L_4_in_L_3/in2');

net = connectLayers(net,'level_3-upsampling','out_L_3_in_L_2/in1');
net = connectLayers(net,'level_2_extract_pyr','out_L_3_in_L_2/in2');

net = connectLayers(net,'level_2-upsampling','out_L_2_in_L_1/in1');
net = connectLayers(net,'level_1_extract_pyr','out_L_2_in_L_1/in2');

net = connectLayers(net,'level_4-upsampling',...
    'level_3-reconstructLayer/in2');
net = connectLayers(net,'level_3-upsampling',...
    'level_2-reconstructLayer/in2');
net = connectLayers(net,'level_2-upsampling',...
    'level_1-reconstructLayer/in2');

net = connectLayers(net,'out_L_2_in_L_1',...
    'level_1-Encoder-Stage-1-Conv-1');
net = connectLayers(net,'out_L_3_in_L_2',...
    'level_2-Encoder-Stage-1-Conv-1');
net = connectLayers(net,'out_L_4_in_L_3',...
    'level_3-Encoder-Stage-1-Conv-1');
net = connectLayers(net,'level_4_extract_pyr',...
    'level_4-Encoder-Stage-1-Conv-1');

net  = dlnetwork(net);
end
