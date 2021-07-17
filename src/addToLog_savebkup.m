%% training progress
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function stop = addToLog_savebkup(info,out_dir, check_point_period)

stop = false;
persistent key_


epoch=info.Epoch;
iter=info.Iteration;
loss_G=gather(info.TrainingLoss_G);
loss_D=gather(info.TrainingLoss_D);
loss_D_R=gather(info.TrainingLoss_D_R);
loss_D_F=gather(info.TrainingLoss_D_F);
lrate=info.BaseLearnRate;
val_loss_G = gather(info.ValidationLoss_G);
val_loss_D = gather(info.ValidationLoss_D);
val_loss_D_R = gather(info.ValidationLoss_D_R);
val_loss_D_F = gather(info.ValidationLoss_D_F);

M=[epoch,iter,loss_G,loss_D,loss_D_R,loss_D_F,val_loss_G,val_loss_D, ...
    val_loss_D_R,val_loss_D_F,lrate];
if info.State == "start"
    key_=char(datetime); key_=strrep(key_,':','-');
    dlmwrite(fullfile(out_dir,sprintf('report_%s.csv',key_)),...
        M,'delimiter',',');
else
    dlmwrite(fullfile(out_dir,sprintf('report_%s.csv',key_)),...
        M,'delimiter',',','-append');
   files=dir(fullfile(out_dir,'G_*'));
    [~,idx] = sort([files.datenum]);
    files=files(idx);
    if mod(epoch,check_point_period)==0
        if exist(fullfile(out_dir,'backup'),'dir') == 0
            mkdir(fullfile(out_dir,'backup'));
        end
        copyfile(fullfile(out_dir,files(end).name),...
            fullfile(out_dir,'backup',sprintf('G_epoch_number_%d.mat',floor(epoch))));
    end
    if length(files)>check_point_period
        for i=1:length(files)-5
            delete(fullfile(out_dir,files(i).name));
        end
    end
    
    files=dir(fullfile(out_dir,'D_*'));
    if isempty(files) == 0
        [~,idx] = sort([files.datenum]);
    files=files(idx);
    if mod(epoch,check_point_period)==0
        if exist(fullfile(out_dir,'backup'),'dir') == 0
            mkdir(fullfile(out_dir,'backup'));
        end
        copyfile(fullfile(out_dir,files(end).name),...
            fullfile(out_dir,'backup',sprintf('D_epoch_number_%d.mat',floor(epoch))));
    end
    if length(files)>check_point_period
        for i=1:length(files)-5
            delete(fullfile(out_dir,files(i).name));
        end
    end
    end
end
end