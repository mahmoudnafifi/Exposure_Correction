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

function [] = runOnFilenames(prefix, ...
    input_ds_filename, output_ds_filename, ...
    input_fs_filename, output_fs_gt_filename)

input_fs = im2double(imread(fullfile(prefix, input_fs_filename)));
edge_fs = rgb2luminance(input_fs); % Used to slice at full resolution.
input_ds = im2double(imread(fullfile(prefix, input_ds_filename)));
edge_ds = rgb2luminance(input_ds); % Determines grid z at low resolution.
output_ds = im2double(imread(fullfile(prefix, output_ds_filename)));
output_fs_gt = im2double(imread(fullfile(prefix, output_fs_gt_filename))); % Ground truth.

% Call driver function.
% [] is for weight_ds: we're not doing a weighted fit.
% grid_size, lambda_s, intensity_options: just use defaults.
results = testBGU(input_ds, edge_ds, output_ds, [], ...
    input_fs, edge_fs, output_fs_gt);
showTestResults(results);
