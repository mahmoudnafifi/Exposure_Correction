%% computes final losses
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%
function [lossGenerator, lossDiscriminator, lossReal, lossGenerated] = ...
    final_Loss(dlY,dlYPredGenerated, dlYPred_D,dlYPredGenerated_D)
if nargin == 2
    lossGenerator = sum(abs(dlY(:)- dlYPredGenerated(:)))/size(dlY,4);
    lossDiscriminator = [];
    lossReal = [];
    lossGenerated = [];
else
    % Calculate losses for the discriminator network.
    lossGenerated = -mean(log(max(1-sigmoid(dlYPredGenerated_D),10^-9)));
    lossReal = -mean(log(max(sigmoid(dlYPred_D),10^-9)));
    
    % Combine the losses for the discriminator network.
    lossDiscriminator = lossReal + lossGenerated;
    
    % Calculate the loss for the generator network.
    W = size(dlY,1)* size(dlY,2) * 12;
    lossGenerator = sum(abs(dlY(:)- dlYPredGenerated(:)))/size(dlY,4) - ...
        W * mean(log(max(sigmoid(dlYPredGenerated_D),10^-9)));
    
end
end