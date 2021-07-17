%% training code
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

clc
clear;
close all;

lR = 10^-4; % initial learning rate

chnls = 16; % number of channels of 1st layere of the encoder for the highest pyramid level

convvfilter = 3; % conv kernel size

encoderDecoderDepth = 3; % numbere of layers (i.e., levels) for the highest pyramid level 

trainingImgsNum = 0; %if 0, then load all training images

withDiscriminator = 1; % include discriminator loss term?

for ps = [128, 256, 512] % for each patch size, do
    
    % please, update training/validation directories accordingly
    
    In_Tr_datasetDir = fullfile('exposure_dataset','training',sprintf('INPUT_IMAGES_P_%d',ps)); % input training patches with size ps size
    
    GT_Tr_datasetDir = fullfile('exposure_dataset','training',sprintf('GT_IMAGES_P_%d',ps)); % ground truth training patches with size ps
    
    In_Vl_datasetDir = fullfile('exposure_dataset','validation',sprintf('INPUT_IMAGES_P_%d',ps)); % validation
    
    GT_Vl_datasetDir = fullfile('exposure_dataset','validation',sprintf('GT_IMAGES_P_%d',ps));
     
    patchSize = [ps, ps, 12]; % 3 color channels x 4 pyramid levels

    switch ps
        
        case 128
            
            dropRate = 20; % drop learning rate
            
            checkpoint_period = 10; % bkup every checkpoint_period
            
            epochs = 40; % number of epochs
            
            miniBatch = 32; % mini-batch size
            
            chkpoint = ''; % start training from scratch -- no chkpoint
            
             if withDiscriminator == 1
                 
                 chkpoint_d = '';
             
             end
            
            validationImgsNum = 2000; % number of validation patches
            
            vlFreq = 5612 *2; % every vlFreq iterations, do validation
            
        case 256
            
            dropRate = 10;
        
            checkpoint_period = 5;
            
            epochs = 30;
            
            miniBatch = 8;
            
            chkpoint = sprintf('model_%d.mat',ps/2);
            
            if withDiscriminator == 1
                chkpoint_d =  '';
            end
            
            validationImgsNum = 1000;
            
            vlFreq = 13230 *2;
        
        case 512
        
            dropRate = 5;
            
            checkpoint_period = 5;
            
            epochs = 20;
            
            miniBatch = 4;
            
            chkpoint = sprintf('model_%d.mat',ps/2);
            
            if withDiscriminator == 1
                chkpoint_d =  sprintf('D_model_%d.mat',ps/2);
            end
            
            validationImgsNum = 500;
            
            vlFreq = 17378 *2;
        
        otherwise
            
            error('wrong ps value');
    end
   checkpoint_dir = sprintf('%dx%d_reports_and_backup_%s',ps,ps,date);
    
    GPUDevice = 1;
    
  
    modelName = sprintf('model_%d.mat',ps);
    
    if withDiscriminator == 1
        D_modelName = sprintf('D_model_%d.mat',ps);
    end
    
    fprintf('Preparing training data ...\n');
    
    [Trdata,Vldata] = getTr_Vl_data(In_Tr_datasetDir, GT_Tr_datasetDir, ...
        In_Vl_datasetDir, GT_Vl_datasetDir, trainingImgsNum, ...
        validationImgsNum, patchSize(1:2),...
        miniBatch);
    
    options = get_trainingOptions(epochs,miniBatch,lR,...
        checkpoint_dir,Vldata,GPUDevice, checkpoint_period, ...
        vlFreq, dropRate);
    
    
    if strcmp(chkpoint,'')
        fprintf('Creating the generator model ...\n');
        net = create_generator(patchSize, encoderDecoderDepth, chnls, convvfilter);
    else
        fprintf('Loading the generator model ...\n');
        load(chkpoint);
        inLayer = imageInputLayer(patchSize,'Name','InputLayer',...
            'Normalization','none');
        net = layerGraph(net);
        net=replaceLayer(net,'InputLayer',inLayer);
        net = dlnetwork(net);
    end
    
    %define/load the discriminator
    if withDiscriminator == 1
        if strcmp(chkpoint_d,'')
            fprintf('Creating the discriminator model ...\n');
            [D] = createDiscriminator();
        else
            fprintf('Loading the discriminator model ...\n');
            load(chkpoint_d);
        end
    end
    
    
    fprintf('Starting training ...\n');
    
    if withDiscriminator == 1
        switch ps
            case 128
                [net, D] = train_network(Trdata,net,[], options);
            case 256
                [net, D] = train_network(Trdata,net,D, options,15);
            case 512
                [net, D] = train_network(Trdata,net,D, options,5);
        end
    else
        [net, ~] = train_network(Trdata,net,[], options);
    end
    
    disp('Done!');
    
    disp('Saving model!');
    
    save(modelName,'net','-v7.3');
    if withDiscriminator == 1
        save(D_modelName,'D','-v7.3');
    end
end




