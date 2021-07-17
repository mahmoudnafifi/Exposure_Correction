%% computes GAN loss (discriminator loss and main network loss)
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%
function [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated)

% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1-sigmoid(dlYPredGenerated)));
lossReal = -mean(log(sigmoid(dlYPred)));

% Combine the losses for the discriminator network.
lossDiscriminator = lossReal + lossGenerated;

% Calculate the loss for the generator network.
lossGenerator = -mean(log(sigmoid(dlYPredGenerated)));

end