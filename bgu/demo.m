% Copyright 2016 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http ://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

%% Local Laplacian on Parrot.
runOnFilenames('../../images/', ...
    'source_low.png', 'target_low.png', ...
    'source.png', 'target.png');

% fprintf('Program paused, press Enter to continue.\n');
% pause;
% close all;
% 
% %% Local Laplacian on Parrot (grayscale).
% runOnFilenames('../../images/', ...
%     'low_res_in.png', 'low_res_out_gray.png', ...
%     'high_res_in.png', 'high_res_ground_truth_out_gray.png');
% 
% fprintf('Program paused, press Enter to continue.\n');
% pause;
% close all;
