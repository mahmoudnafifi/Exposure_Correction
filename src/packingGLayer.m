%% custom network layer to append ground truth pyramid leves together
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

classdef packingGLayer < nnet.layer.Layer
    
    properties
      
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = packingGLayer(name)
            layer.Name = name;
            layer.Description = "Packing Gaussian pyr layer";
            layer.NumInputs = 4;
        end
        
        function Z = predict(layer, X1, X2, X3, X4)
            sz = size(X1);
            L = sz(3)/4;
            
            if length(size(X1)) == 4
                Z = zeros(size(X1,1),size(X1,2),size(X1,3) * 4,size(X4,4),...
                    'like',X1);
                Z(:,:,1:3,:) = X1;
                Z(1:size(X2,1),1:size(X2,2),4:6,:) = X2;
                Z(1:size(X3,1),1:size(X3,2),7:9,:) = X3 * 2;
                Z(1:size(X4,1),1:size(X4,2),10:12,:) = X4 * 2 * 2;
                
            else
                Z = zeros(size(X1,1),size(X1,2),size(X1,3) * 4,...
                    'like',X1);
                Z(:,:,1:3) = X1;
                Z(1:size(X2,1),1:size(X2,2),4:6) = X2;
                Z(1:size(X3,1),1:size(X3,2),7:9) = X3 *2;
                Z(1:size(X4,1),1:size(X4,2),10:12) = X4 * 2 * 2;
            end
        end
        
        
    end
end