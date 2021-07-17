%% custom network layere to extract pyramid level
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

classdef extractPyrLayer < nnet.layer.Layer
    
    properties
        Plevel
    end
    
    properties (Learnable)
    end
    
    methods
        function layer = extractPyrLayer(name, Plevel)
            layer.Name = name;
            layer.Description = "extract lap pyr layer";
            layer.Plevel = Plevel;
        end
        
        function Z = predict(layer, X)
            sz = size(X);
            L = sz(3)/4;
            if length(size(X)) == 4
                Z = X(1:sz(1)/2^(layer.Plevel-1), 1:sz(2)/2^(layer.Plevel-1),...
                    1+ L *(layer.Plevel-1) : L*(layer.Plevel-1) + L,:);
         
            else
                Z = X(1:sz(1)/2^(layer.Plevel-1), 1:sz(2)/2^(layer.Plevel-1),...
                    1+ L *(layer.Plevel-1) : L*(layer.Plevel-1) + L);
            end
        end
        
        
    end
end