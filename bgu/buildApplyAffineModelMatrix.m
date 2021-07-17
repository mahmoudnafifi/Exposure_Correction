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

function A = buildApplyAffineModelMatrix(input_image, num_output_channels)

num_pixels = size(input_image, 1) * size(input_image, 2);

% TODO: vectorize or output the triplet format.
% Then use horzcat quickly concat the cell array.
A = sparse(0, 0);
for k = 1:size(input_image,3)
    plane = input_image(:,:,k);
    % Repeat each component num_output_channels times.
    sd = spdiags(repmat(plane(:), [num_output_channels 1]), 0, ...
        num_output_channels * num_pixels, num_output_channels * num_pixels);
    A = [A sd];
end

% Ones channel.
ones_diag = spdiags(ones(num_output_channels * num_pixels, 1), 0, ...
    num_output_channels * num_pixels, num_output_channels * num_pixels);

A = [A ones_diag];
