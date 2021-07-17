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

function grid_size = getDefaultAffineGridSize(input_image, output_image)

input_height = size(input_image, 1);
input_width = size(input_image, 2);
input_channels = size(input_image, 3);
output_channels = size(output_image, 3);

grid_size = [22, 22, 8, output_channels, input_channels + 1];%round([input_height / 12, input_width / 12, 12, ...
   % output_channels, input_channels + 1]);
