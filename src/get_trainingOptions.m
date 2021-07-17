%% training settings
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function options = get_trainingOptions(epochs,miniBatch,lR,...
    checkpoint_dir,validation_data,GPUDevice, checkpoint_period, ...
    vlFrequencey,dropRate)


gpuDevice(GPUDevice);

if exist(checkpoint_dir,'dir') == 0
    mkdir(checkpoint_dir);
end



options.InitialLearnRate = lR;
options.MaxEpochs = epochs;
options.MiniBatchSize = miniBatch;
options.CheckpointPath = checkpoint_dir;
options.Shuffle = 'every-epoch';%'once', ...
options.VerboseFrequency = 50;
options.checkpoint_period = checkpoint_period;
options.LearnRateDropPeriod = dropRate;
options.LearnRateDropFactor = 0.5;
options.Plots = 'training-progress';
options.ExecutionEnvironment = 'gpu';
options.GradientDecayFactor = 0.9;
options.SquaredGradientDecayFactor= 0.999;
options.ValidationData = validation_data;
options.ValidationFrequency = vlFrequencey;
