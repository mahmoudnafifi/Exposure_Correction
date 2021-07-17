%
I_1 = im2double(imread(fullfile('HDR_images','1.jpg')));
I_2 = im2double(imread(fullfile('HDR_images','2.jpg')));
I_3 = im2double(imread(fullfile('HDR_images','3.jpg')));



I = zeros(size(I_1,1),size(I_1,2),size(I_1,3),3);
I(:,:,:,1) = flip(flip(I_1,2));
I(:,:,:,2) = flip(flip(I_2,2));
I(:,:,:,3) = flip(flip(I_3,2));

[R,W] = exposure_fusion(I,[1 1 1]);

imwrite(R,fullfile('HDR_images','HDR_fusion.jpg'));

save(fullfile('HDR_images','W.mat'),'W');