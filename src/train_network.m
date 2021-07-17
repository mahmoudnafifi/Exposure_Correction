%% training loop
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function [dlnetGenerator,dlnetDiscriminator] = ...
    train_network(Trdata,dlnetGenerator,dlnetDiscriminator, options, T)

if nargin == 4
    T = 10;
end

if isempty(dlnetDiscriminator) == 1
    wD = 0;
else
    wD = 1;
end

% fetching training parameters from 'options'
numEpochs = options.MaxEpochs;
miniBatchSize = options.MiniBatchSize;
Trdata.MiniBatchSize = miniBatchSize;
dropRate = options.LearnRateDropPeriod;
dropFactor = options.LearnRateDropFactor;
learnRateGenerator = options.InitialLearnRate;


if wD == 1
    learnRateDiscriminator = options.InitialLearnRate/10;
end
VerboseFrequency = options.VerboseFrequency;
checkpoint_dir = options.CheckpointPath;
checkpoint_period = options.checkpoint_period;
vlData = options.ValidationData;
if isempty(vlData)
    validation_process = 0;
else
    validation_process = 1;
end
shuffling = options.Shuffle;
vlFreq = options.ValidationFrequency;
gradientDecayFactor = options.GradientDecayFactor;
squaredGradientDecayFactor = options.SquaredGradientDecayFactor;
executionEnvironment = options.ExecutionEnvironment;
plot_training = options.Plots;

% initialization
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
if wD == 1
    trailingAvgDiscriminator = [];
    trailingAvgSqDiscriminator = [];
end

if strcmpi(plot_training,'training-progress') == 1
    figure
    
    lineLossTrain_G = animatedline('Color', 'blue');
    if wD == 1
        lineLossTrain_D_F = animatedline('Color', 'red');
        lineLossTrain_D_R = animatedline('Color', 'green');
    end
    
    lineLossValidate_G = animatedline('Color', [0.5,0.5,0.5],...
        'LineWidth',1.7,'Marker','o','LineStyle' ,'--');
    
    addpoints(lineLossTrain_G,0,0);
    if wD == 1
        addpoints(lineLossTrain_D_F,0,0);
        addpoints(lineLossTrain_D_R,0,0);
    end
    
    addpoints(lineLossValidate_G,0,0);
    
    
    if wD == 1
        legend('G loss','D loss (generated)', 'D loss (real)','Val G loss');
    else
        legend('G loss','Val G loss');
    end
    
    xlabel("Iteration")
    ylabel("Loss")
end
iteration = 0;
start = tic;
initial_logging = 1;
last_loss_G = 0;
if wD == 1
    last_loss_D_F = 0;
    last_loss_D_R = 0;
end
% Loop over epochs.
for i = 1 :numEpochs
    % Reset and shuffle datastore.
    
    reset(Trdata);
    if strcmpi(shuffling,'once') == 1 && i == 1
        Trdata = shuffle(Trdata);
    elseif strcmpi(shuffling,'every-epoch') == 1
        Trdata = shuffle(Trdata);
    end
    
    mean_loss_G = last_loss_G;
    if wD == 1
        if i>T%mod(i-1,2) ~= 0
            mean_loss_D_F = last_loss_D_F;
            mean_loss_D_R = last_loss_D_R;
        else
            mean_loss_D_R = 0;
            mean_loss_D_F = 0;
        end
    end
    
    % Loop over mini-batches.
    while hasdata(Trdata)
        
        iteration = iteration + 1;
        
        aug_ind = randi(2);
        save('augInd.mat','aug_ind');
        
        
        % Read mini-batch of data.
        data = read(Trdata);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data
        X = cat(4,data{:,1}{:});
        Y = cat(4,data{:,2}{:});
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlY = dlarray(Y, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || ...
                executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end
        
        % Evaluate the model gradients and the generator state
        if i>T && wD == 1%mod(i-1,2) == 0 && wD == 1
            [gradientsGenerator, gradientsDiscriminator, lossG,...
                lossR, lossF, stateGenerator] = ...
                dlfeval(@modelGradients, dlnetGenerator, ...
                dlnetDiscriminator,dlX, dlY,1);
            dlnetGenerator.State = stateGenerator;
            
            % Update the discriminator network parameters.
            [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,...
                trailingAvgSqDiscriminator] = ...
                adamupdate(dlnetDiscriminator.Learnables, ...
                gradientsDiscriminator, ...
                trailingAvgDiscriminator, ...
                trailingAvgSqDiscriminator, iteration, ...
                learnRateDiscriminator, gradientDecayFactor,...
                squaredGradientDecayFactor);
            
            % Update the generator network parameters.
            [dlnetGenerator.Learnables,trailingAvgGenerator,...
                trailingAvgSqGenerator] = ...
                adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
                trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
                learnRateGenerator, gradientDecayFactor, ...
                squaredGradientDecayFactor);
            
            loss_g_extracted = double(gather(extractdata(lossG)))/...
                (size(dlX,1)*size(dlX,2) * 3 * 255);
            loss_d_extracted_real = double(gather(extractdata(lossR)));
            loss_d_extracted_fake = double(gather(extractdata(lossF)));
            
            mean_loss_G = mean_loss_G + loss_g_extracted;
            mean_loss_D_R = mean_loss_D_R + loss_d_extracted_real;
            mean_loss_D_F = mean_loss_D_F + loss_d_extracted_fake;
        else
            [gradientsGenerator, ~, lossG,...
                ~, ~, stateGenerator] = ...
                dlfeval(@modelGradients, dlnetGenerator, ...
                dlnetDiscriminator,dlX, dlY,0);
            dlnetGenerator.State = stateGenerator;
            
            
            % Update the generator network parameters.
            [dlnetGenerator.Learnables,trailingAvgGenerator,...
                trailingAvgSqGenerator] = ...
                adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
                trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
                learnRateGenerator, gradientDecayFactor, ...
                squaredGradientDecayFactor);
            
            loss_g_extracted = double(gather(extractdata(lossG)))/...
                (size(dlX,1)*size(dlX,2) * 3 * 255);
            
            
            mean_loss_G = mean_loss_G + loss_g_extracted;
            
        end
        if mod(iteration,VerboseFrequency) == 0
            mean_loss_G = mean_loss_G/VerboseFrequency;
            if  i>T && wD == 1 %mod(i-1,2) == 0 && wD == 1
                mean_loss_D_R = mean_loss_D_R/VerboseFrequency;
                mean_loss_D_F = mean_loss_D_F/VerboseFrequency;
            end
            
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            
            if wD == 1
                fprintf("Epoch: " + i + "|" + iteration + ...
                    ", elapsed: " + string(D) + ", G loss:" + ...
                    num2str(mean_loss_G) + ...
                    ", D loss (real):" + num2str(mean_loss_D_R) + ...
                    ", D loss (fake):" + num2str(mean_loss_D_F) + ...
                    "\n");
            else
                fprintf("Epoch: " + i + "|" + iteration + ...
                    ", elapsed: " + string(D) + ", G loss:" + ...
                    num2str(mean_loss_G) + ...
                    "\n");
            end
            
            if initial_logging == 1
                info.State = "start";
                initial_logging = 0;
            else
                info.State = "already started";
            end
            
            info.Epoch = i;
            info.TrainingLoss_G = mean_loss_G;
            if wD == 1
                info.TrainingLoss_D = mean_loss_D_R + mean_loss_D_F;
                info.TrainingLoss_D_R = mean_loss_D_R;
                info.TrainingLoss_D_F = mean_loss_D_R;
            else
                info.TrainingLoss_D = [];
                info.TrainingLoss_D_R = [];
                info.TrainingLoss_D_F = [];
            end
            info.Iteration = iteration;
            info.BaseLearnRate = learnRateGenerator;
            info.ValidationLoss_G = [];
            info.ValidationLoss_D = [];
            info.ValidationLoss_D_R = [];
            info.ValidationLoss_D_F = [];
            
            stop = addToLog_savebkup(info,checkpoint_dir,checkpoint_period);
            
            if stop == true
                return;
            end
            
            if strcmpi(plot_training,'training-progress') == 1
                addpoints(lineLossTrain_G,iteration,double(mean_loss_G))
                if wD == 1
                    addpoints(lineLossTrain_D_R,iteration,...
                        double(mean_loss_D_R))
                    addpoints(lineLossTrain_D_F,iteration,...
                        double(mean_loss_D_F))
                end
                title("Epoch: " + i + ", elapsed: " + string(D))
                drawnow
            end
            last_loss_G = mean_loss_G;
            mean_loss_G = 0;
            if i>T && wD == 1 %mod(i-1,2) == 0 && wD == 1
                last_loss_D_R = mean_loss_D_R;
                last_loss_D_F = mean_loss_D_F;
                mean_loss_D_R = 0;
                mean_loss_D_F = 0;
            end
            
        end
        
        if validation_process == 1
            if mod(iteration,vlFreq) == 0 || iteration == 1
                
                %% validation code
                % Loop over mini-batches.
                iteration_v = 0;
                lossG_val = 0;
                if wD == 1
                    lossR_val = 0;
                    lossF_val = 0;
                end
                reset(vlData);
                while hasdata(vlData)
                    iteration_v = iteration_v + 1;
                    
                    % Read mini-batch of data.
                    data = read(vlData);
                    
                    % Ignore last partial mini-batch of epoch.
                    if size(data,1) < miniBatchSize
                        continue
                    end
                    
                    % Concatenate mini-batch of data
                    X = cat(4,data{:,1}{:});
                    Y = cat(4,data{:,2}{:});
                    
                    % Convert mini-batch of data to dlarray specify the dimension labels
                    % 'SSCB' (spatial, spatial, channel, batch).
                    dlX = dlarray(X, 'SSCB');
                    dlY = dlarray(Y, 'SSCB');
                    
                    % If training on a GPU, then convert data to gpuArray.
                    if (executionEnvironment == "auto" && canUseGPU) || ...
                            executionEnvironment == "gpu"
                        dlX = gpuArray(dlX);
                        dlY = gpuArray(dlY);
                    end
                    
                    % Evaluate the model gradients and the generator state
                    if wD == 1
                        [~, ~, lossG, lossR, lossF, ~] = ...
                            dlfeval(@modelGradients, dlnetGenerator, ...
                            dlnetDiscriminator, dlX, dlY, 1, 1);
                        lossG_val = lossG_val + gather(extractdata(lossG));
                        lossR_val = lossR_val + gather(extractdata(lossR));
                        lossF_val = lossF_val + gather(extractdata(lossF));
                    else
                        [~, ~, lossG, ~, ~, ~] = ...
                            dlfeval(@modelGradients, dlnetGenerator, ...
                            dlnetDiscriminator, dlX, dlY, 0, 1);
                        lossG_val = lossG_val + gather(extractdata(lossG));
                    end
                    
                end
                loss_g_extracted_val = lossG_val/(iteration_v * ...
                    size(dlX,1)*size(dlX,2) * 3 * 255);
                if wD == 1
                    loss_d_extracted_val = (lossR_val+lossF_val)/...
                        iteration_v;
                    loss_d_extracted_real_val = lossR_val/iteration_v;
                    loss_d_extracted_fake_val  = lossF_val/iteration_v;
                else
                    loss_d_extracted_val = [];
                    loss_d_extracted_real_val = [];
                    loss_d_extracted_fake_val =[];
                end
                if initial_logging == 1
                    info.State = "start";
                    initial_logging = 0;
                else
                    info.State = "already started";
                end
                info.Epoch = i;
                
                info.TrainingLoss_G = [];
                info.TrainingLoss_D = [];
                info.TrainingLoss_D_R = [];
                info.TrainingLoss_D_F = [];
                
                info.Iteration = iteration;
                info.BaseLearnRate = learnRateGenerator;
                info.ValidationLoss_G = loss_g_extracted_val;
                info.ValidationLoss_D = loss_d_extracted_val;
                info.ValidationLoss_D_R = loss_d_extracted_real_val;
                info.ValidationLoss_D_F = loss_d_extracted_fake_val;
                
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                if wD == 1
                    fprintf("Validation results:\nEpoch: " + i + "|" + ...
                        iteration + ", Elapsed: " + string(D) + ...
                        ", val. G loss:" + ...
                        num2str(loss_g_extracted_val)+ ...
                        ", val. D loss:" + ...
                        num2str(loss_d_extracted_val) + "\n");
                else
                    fprintf("Validation results:\nEpoch: " + i + "|" + ...
                        iteration + ", Elapsed: " + string(D) + ...
                        ", val. G loss:" + ...
                        num2str(loss_g_extracted_val)+ "\n");
                end
                
                if strcmpi(plot_training,'training-progress') == 1
                    addpoints(lineLossValidate_G,iteration,...
                        double(loss_g_extracted_val))
                    title("Epoch: " + i + ", elapsed: " + string(D))
                    drawnow
                end
                stop = addToLog_savebkup(info,checkpoint_dir,...
                    checkpoint_period);
                
                if stop == true
                    return;
                end
            end
        end
    end
    if mod(i,dropRate) == 0
        if wD == 1
            learnRateDiscriminator = learnRateDiscriminator * dropFactor;
        end
        learnRateGenerator = learnRateGenerator * dropFactor;
    end
    save(fullfile(checkpoint_dir,sprintf('G_model_ep_%d.mat',i)),...
        'dlnetGenerator','-v7.3');
    if wD == 1
        save(fullfile(checkpoint_dir,sprintf('D_model_ep_%d.mat',i)),...
            'dlnetDiscriminator','-v7.3');
    end
end

