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

% Bilateral Guided Upsampling fits an affine bilateral grid gamma for an
%   operator f: input_image --> output_image.
%
% function [gamma, A, b, lambda_spatial, intensity_options] =
%   bguFit(input_image, edge_image, ...
%   output_image, output_weight, grid_size, lambda_spatial, intensity_options)
%
% Inputs:
%
% input_image is a double tensor: the input to the operator f.
%   Dimensions: height x width x num_input_channels
% edge_image is a double matrix: the "edges" we try to respect. For
%   example, the input luminance (try rgb2luminance or rgb2gray).
%   Dimensions: height x width (x 1)
% output_image is a double tensor: the output f(input_image).
%   Dimensions: height x width x num_output_channels
%
% [Optional] weight_image is a double tensor: whether f is defined at each
%   pixel. Weights need only be non-negative (>= 0) and can be any double.
%   Dimensions: height x width x num_output_channels
%   (same size as output_image)
%
% [Optional] grid_size is:
%   [height width depth num_output_channels num_input_channels]
%   If not specified or empty, defaults to:
%   [round(input_height / 16), round(input_width / 16), 8, ...
%    output_channels, input_channels + 1]
%
% [Optional] lambda_spatial controls spatial smoothness, and defaults to 1.
%   lambda_spatial must be positive.
%
% [Optional] intensity_options is a struct:
%   .type:
%     The type of constraint you want on the grid in the intensity
%     direction. Must be one of:
%       'none': no constraints
%       'first': constrain the first derivative dg/dz
%       'second': constrain the second derivative d2g/dz2
%     Default: 'second'
%   .value:
%     The value towards which the first or second derivative should be.
%     Note that this value is in *absolute* output units and is independent
%     of the grid spacing. I.e., if you want your derivatives to be close
%     to 1 in each bin, then set type to 'first' and value to 1, not
%     1/grid_size(3).
%       type = 'none': ignored.
%       type = 'first' or 'second': any double.
%     Default: 0
%   .lambda:
%     The strength of the constraint relative to the other terms.
%     Default: 4e-6 for first derivative, 4e-7 for second derivative.
%
% Outputs:
%
% A, b are the optional output sparse matrix and right hand side vector
% used to solve for gamma. gamma = reshape(A \ b, grid_size).
%
% lambda_spatial and intensity_options are echoed back as optional outputs
% so the test harness can ask for default parameters then retrieve them.
function [gamma, A, b, lambda_spatial, intensity_options] = ...
    bguFit(input_image, edge_image, ...
    output_image, output_weight, grid_size, lambda_spatial, intensity_options)

DEFAULT_LAMBDA_SPATIAL = 1;

DEFAULT_FIRST_DERIVATIVE_LAMBDA_Z = 4e-6; % about 0.01 for default bin sizes.
%DEFAULT_FIRST_DERIVATIVE_LAMBDA_Z = 4e-7; % about 0.001 for default bin sizes.

DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z = 4e-7; % about 0.01 for default bin sizes.
%DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z = 4e-8; % about 0.001 for default bin sizes.

if ~isa(input_image, 'double')
    error('input_image must be double.');
end

if ~isa(output_image, 'double')
    error('model_image must be double.');
end

if ~exist('edge_image', 'var') || isempty(edge_image) || ...
        ~ismatrix(edge_image) || ~isa(edge_image, 'double')
    error('edge_image must be a double matrix (one channel).');
end

if isempty(output_weight)
    output_weight = ones(size(output_image));
end

if ~isa(output_weight, 'double')
    error('weight must be double matrix');
end

if ndims(input_image) < 2
    error('input_image must be at least two-dimensional.');
end

if ndims(output_image) < 2
    error('output_image must be at least two-dimensional.');
end

if ~isequal(size(input_image, 1), size(output_image, 1)) || ...
    ~isequal(size(input_image, 2), size(output_image, 2))
    error('input_image and output_image must have the same width and height.');
end

if ~exist('grid_size', 'var') || isempty(grid_size)
    grid_size = getDefaultAffineGridSize(input_image, output_image);
end

if ~exist('lambda_spatial', 'var') || isempty(lambda_spatial)
    lambda_spatial = DEFAULT_LAMBDA_SPATIAL;
end

if lambda_spatial <= 0
    error('lambda_spatial must be positive.');
end

% If you pass in nothing, default to second derivative.
if ~exist('intensity_options', 'var') || isempty(intensity_options)
    intensity_options.type = 'second';
    intensity_options.value = 0;
    intensity_options.lambda = DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z;
    intensity_options.enforce_monotonic = false;
end

% If you ask for a constraint but are missing some of the parameters.
if strcmp(intensity_options.type, 'first')
    if ~isfield(intensity_options, 'lambda')
        intensity_options.lambda = DEFAULT_FIRST_DERIVATIVE_LAMBDA_Z;
    end

    if ~isfield(intensity_options, 'value')
        intensity_options.value = 0;
    end

    if ~isfield(intensity_options, 'enforce_monotonic')
        intensity_options.enforce_monotonic = false;
    end
elseif strcmp(intensity_options.type, 'second')
    if ~isfield(intensity_options, 'lambda')

        intensity_options.lambda = DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z;
    end

    if ~isfield(intensity_options, 'value')
        intensity_options.value = 0;
    end

    if ~isfield(intensity_options, 'enforce_monotonic')
        intensity_options.enforce_monotonic = false;
    end
else
    if ~isfield(intensity_options, 'enforce_monotonic')
        intensity_options.enforce_monotonic = false;
    end
end

input_height = size(input_image, 1);
input_width = size(input_image, 2);
grid_height = grid_size(1);
grid_width = grid_size(2);
grid_depth = grid_size(3);
affine_output_size = grid_size(4);
affine_input_size = grid_size(5);

% Size of each grid cell in pixels (# pixels per bin).
bin_size_x = input_width / grid_width;
bin_size_y = input_height / grid_height;
bin_size_z = 1 / grid_depth;

num_deriv_y_rows = (grid_height - 1) * grid_width * grid_depth ...
    * affine_output_size * affine_input_size;
num_deriv_x_rows = grid_height * (grid_width - 1) * grid_depth ...
    * affine_output_size * affine_input_size;

% Set up data term Ax = b.
%
% x is the bilateral grid. It is vectorized as a column vector of size n,
% where n = grid_height * grid_width * grid_depth * ...
%   affine_output_size * affine_input_size.
%
% The right hand side b is the *output* image, vectorized as output_image(:).
% I.e., it's a m x 1 column vector:
% [ red0 red1 ... redN green0 green1 ... greenN blue0 blue1 ... blueN ]'
%
% A is an m x n (sparse) matrix.
% It slices into the bilateral grid with linear interpolation at edge_image
% to get an affine model, then applies it.

% TODO: vectorize
% Build slice matrices for each (i,j) entry of the affine model.
weight_matrices = cell(affine_output_size, affine_input_size);
slice_matrices = cell(affine_output_size, affine_input_size);
for j = 1:affine_input_size    
    for i = 1:affine_output_size
        %fprintf('Building weight and slice matrices, i = %d, j = %d\n', i, j);
        [weight_matrices{i, j}, slice_matrices{i, j}] = ...
            buildAffineSliceMatrix(input_image, edge_image, grid_size, i, j);
    end
end

% Concat them together.
slice_matrix = sparse(0, 0);
weight_matrix = sparse(0, 0);
for j = 1:affine_input_size
    for i = 1:affine_output_size
        %fprintf('Concatenating affine slice matrices, i = %d, j = %d\n', i, j);
        slice_matrix = [slice_matrix; slice_matrices{i, j}];
        weight_matrix = blkdiag(weight_matrix, weight_matrices{i, j});
    end
end

%fprintf('Building apply affine model matrix\n');
apply_affine_model_matrix = buildApplyAffineModelMatrix(...
    input_image, affine_output_size);

% Complete slicing matrix.
%fprintf('Building full slice matrix\n');
sqrt_w = sqrt(output_weight(:)); % weighted least squares takes sqrt(w).
W_data = spdiags(sqrt_w, 0, numel(output_weight), numel(output_weight));
A_data = W_data * apply_affine_model_matrix * weight_matrix * slice_matrix;
b_data = output_image(:) .* sqrt_w;

% ----- Add deriv_y constraints -----
%fprintf('Building d/dy matrix\n');
A_deriv_y = (bin_size_x * bin_size_z / bin_size_y) * lambda_spatial * ...
    buildDerivYMatrix(grid_size);
b_deriv_y = zeros(num_deriv_y_rows, 1);

% ----- Add deriv_x constraints -----
%fprintf('Building d/dx matrix\n');
A_deriv_x = (bin_size_y * bin_size_z / bin_size_x) * lambda_spatial * ...
    buildDerivXMatrix(grid_size);
b_deriv_x = zeros(num_deriv_x_rows, 1);

% ----- Add intensity constraints -----
%fprintf('Building d/dz matrix\n');
if strcmp(intensity_options.type, 'first')
    A_intensity = (bin_size_x * bin_size_y / bin_size_z) * ...
        intensity_options.lambda * buildDerivZMatrix(grid_size);
    value = intensity_options.lambda * intensity_options.value;
    m = size(A_intensity, 1);
    b_intensity = value * ones(m, 1);
elseif strcmp(intensity_options.type, 'second')
    A_intensity = (bin_size_x * bin_size_y) / (bin_size_z * bin_size_z) * ...
        intensity_options.lambda * ...
        buildSecondDerivZMatrix(grid_size);
    value = intensity_options.lambda * intensity_options.value;
    m = size(A_intensity, 1);
    b_intensity = value * ones(m, 1);
end

% ----- Assemble final A matrix -----
%fprintf('Assembling final sparse system\n');
if ~strcmp(intensity_options.type, 'none')
    A = [A_data; A_deriv_y; A_deriv_x; A_intensity];
    b = [b_data; b_deriv_y; b_deriv_x; b_intensity];
else
    A = [A_data; A_deriv_y; A_deriv_x];
    b = [b_data; b_deriv_y; b_deriv_x];
end

% ----- Solve -----
%fprintf('Solving system\n');
gamma = A \ b;
gamma = reshape(gamma, grid_size);
