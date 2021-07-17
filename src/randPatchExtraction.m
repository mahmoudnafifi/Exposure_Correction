%% main function to extract random patches with different dimensions
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function randPatchExtraction(in_dir,gt_dir,...
    out_dir, gt_out_dir, Psize, Pnum_per_img)

if exist(out_dir,'dir') == 0
    mkdir(out_dir);
end

if exist(gt_out_dir,'dir') == 0
    mkdir(gt_out_dir);
end

images = dir(fullfile(gt_dir,'*.jpg'));

images = {images(:).name};

for i = 1 : length(images)
    fprintf('[%dx%d]: Processing %s (%d/%d)... \n', Psize(1),Psize(2),...
        images{i},i,length(images));
    [~,name,ext] = fileparts(images{i});
    In_images = dir(fullfile(in_dir,[name '*']));
    In_images = {In_images(:).name};
    I = im2double(imread(fullfile(gt_dir,images{i})));
    sz = size(I);
    count = 0;
    trials = 0;
    while 1
        if trials >=200
            break;
        end
        if sz(1)<= Psize(1) || sz(2)<= Psize(2)
            break;
        end
        r1 = randi(sz(1)-Psize(1));
        r2 = randi(sz(2)-Psize(2));
        if r1 + Psize(1)-1 > sz(1) || r2 + Psize(2)-1 > sz(2)
            trials = trials + 1;
            continue;
        end
        Gpatch = I(r1:r1 + Psize(1)-1,r2: r2 + Psize(2)-1,:);
        if sum(Gpatch(:))/(Psize(1)*Psize(2)) < 0.02 || ...
                sum(Gpatch(:))/(Psize(1)*Psize(2)*3) > 0.98
            trials = trials + 1;
            continue;
        end
        
        Gmag = imgradient(rgb2gray(Gpatch));
        if sum(Gmag(:))/length(Gmag(:)) < 0.06
            trials = trials + 1;
            continue;
        end
        count = count + 1;
        imwrite(Gpatch, fullfile(gt_out_dir, [name '_' num2str(count) ext]));
        
        for k = 1 : length(In_images)
            curr_name = In_images{k};
            [~,Tname,ext] = fileparts(curr_name);
            I_in = im2double(imread(fullfile(in_dir,curr_name)));
            Tpatch = I_in(r1:r1 + Psize(1)-1,r2: r2 + Psize(2)-1,:);
            parts = strsplit(Tname,'_');
            imwrite(Tpatch, fullfile(out_dir,[...
                Tname((1:end-length(parts{end})-1)) '_' num2str(count) '_' ...
                parts{end} ext]));
        end
        if count == Pnum_per_img + 2
            break;
        end
    end
end

