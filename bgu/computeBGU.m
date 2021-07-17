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

% Run bilateral guided upsampling pipeline on downsampled input and output, then
% compare to ground truth.
%
% input_ds: downsampled input
% edge_ds: downsampled edge
% output_ds: downsampled output
% weight_ds: downsampled weight map
% input_fs: full-size input
% edge_fs: full-size edge
%
% output_fs: ground-truth full-size output
% [optional] grid_size: affine bilateral grid size to pass through
%   [height width depth num_output_channels num_input_channels]
% [optional] lambda_s: spatial smoothness parameter to pass through
% [optional] intensity_options: intensity parameters to pass through
function output = computeBGU(input_ds, edge_ds, output_ds, weight_ds, ...
    input_fs, edge_fs, grid_size, lambda_s, intensity_options)

output.input_ds = input_ds;
output.edge_ds = edge_ds;
output.output_ds = output_ds;
output.input_fs = input_fs;
output.edge_fs = edge_fs;

if exist('grid_size', 'var')
    output.grid_size = grid_size;
else
    output.grid_size = [];
end

if exist('lambda_s', 'var')
    output.lambda_s = lambda_s;
else
    output.lambda_s = [];
end

if exist('intensity_options', 'var')
    output.intensity_options = intensity_options;
else
    output.intensity_options = [];
end

tic;
[output.gamma, output.A, output.b, output.lambda_s, output.intensity_options] = ...
    bguFit(input_ds, edge_ds, ...
    output_ds, weight_ds, output.grid_size, output.lambda_s, output.intensity_options);
t_fit = toc;

output.grid_size = size(output.gamma);

tic;
output.result_fs = bguSlice(output.gamma, input_fs, edge_fs);
t_slice = toc;

output.t_fit = t_fit;
output.t_slice = t_slice;


