%% computes gradients of our main network (generator) and the discriminator network
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function [gradientsGenerator, gradientsDiscriminator, ...
    lossGenerator, Rloss, Gloss, stateGenerator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlY, calcD, isValidation)

if nargin == 5
    isValidation = 0;
end

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlX);

if calcD == 1
% Calculate the predictions for real data with the discriminator network.
dlY_D = dlarray(imresize(augmentBatch(...
    extractdata(dlY(:,:,1:3,:))/255),[256,256]),'SSCB');
dlYPred = forward(dlnetDiscriminator, dlY_D);

dlXGenerated_D = dlarray(augmentBatch(single(uint8(imresize(...
    extractdata(dlXGenerated(:,:,1:3,:)),[256,256])))/255),'SSCB');
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated_D);

% Calculate the final loss
[lossGenerator, lossDiscriminator, Rloss, Gloss] = ...
    final_Loss(dlY, dlXGenerated, dlYPred, dlYPredGenerated);
else
    % Calculate the final loss
[lossGenerator] = final_Loss(dlY, dlXGenerated);
lossDiscriminator = []; Rloss= []; Gloss = [];
end


if isValidation == 0
% For each network, calculate the gradients with respect to the loss.
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
if calcD == 1
    gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);
else
    gradientsDiscriminator = [];
    
end
else
    gradientsGenerator = [];
    gradientsDiscriminator = [];
    
end

end