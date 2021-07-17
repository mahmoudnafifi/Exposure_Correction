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

% Slice (apply) bilateral guided upsampling.
%
% Inputs:
%
% gamma: the affine bilateral grid.
%   size: [height width depth num_output_channels num_input_channels]
% input_image is a double tensor: the (full-resolution) input image
% edge_image is a double matrix: the (full-resolution) edge image
%
function output = bguSlice(gamma, input_image, edge_image)

% Find downsampling coordinates, without rounding.
input_height = size(input_image, 1);
input_width = size(input_image, 2);
grid_height = size(gamma, 1);
grid_width = size(gamma, 2);
grid_depth = size(gamma, 3);
affine_output_size = size(gamma, 4);
affine_input_size = size(gamma, 5);

% meshgrid inputs and outputs are x, then y, with x right, y down.
[ x, y ] = meshgrid(0:(input_width - 1), 0:(input_height - 1));

% Downsample x and y to grid space (leaving them as floats).
bg_coord_x = ((x + 0.5) * (grid_width - 1) / input_width);
bg_coord_y = ((y + 0.5) * (grid_height - 1) / input_height);
bg_coord_z = edge_image * (grid_depth - 1);

% Add 1 to all coordinates for MATLAB.
bg_coord_xx = bg_coord_x + 1;
bg_coord_yy = bg_coord_y + 1;
bg_coord_zz = bg_coord_z + 1;

% interp3 takes xx, yy, zz.
affine_model = {affine_output_size, affine_input_size};
for j = 1:affine_input_size
    for i = 1:affine_output_size        
            affine_model{i,j} = interp3(gamma(:,:,:,i,j), ...
                bg_coord_xx, bg_coord_yy, bg_coord_zz);
    end
end

% Iterate over each row.
% TODO: optimize this.
for i = 1:affine_output_size
    affine_model2{i,1} = cat(4, affine_model{i,:});
end
affine_model3 = cat(3, affine_model2{:,1});

input1 = cat(3, input_image, ones(size(input_image,1), size(input_image,2)));

output = 0;
for i = 1:affine_input_size
    output = output + bsxfun(@times, affine_model3(:,:,:,i), input1(:,:,i));
end
