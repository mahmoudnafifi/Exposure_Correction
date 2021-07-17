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

% Build the matrices that slices out a particular (i,j) component of the
% affine model stored in a 3D bilateral grid using trilinear interpolation.
function [w, st] = buildAffineSliceMatrix(input_image, edge_image, ...
    grid_size, i, j)

num_grid_cells = prod(grid_size);
image_width = size(input_image, 2);
image_height = size(input_image, 1);
num_pixels = image_width * image_height;
grid_width = grid_size(2);
grid_height = grid_size(1);
grid_depth = grid_size(3);

pixel_x = 0:(size(input_image, 2) - 1);
pixel_y = (0:(size(input_image, 1) - 1))';

% Convert to floating point bilateral grid coordinates:
%   x and y are shifted by a half pixel: pixels are considered to be at the
%   center of their little square.
%   Multiply (grid_width - 1) / image_width: let grid samples be defined
%   at integer edges.
bg_coord_x = (pixel_x + 0.5) * (grid_width - 1) / image_width;
bg_coord_y = (pixel_y + 0.5) * (grid_height - 1) / image_height;
bg_coord_z = edge_image * (grid_depth - 1);

bg_idx_x0 = floor(bg_coord_x);
bg_idx_y0 = floor(bg_coord_y);
bg_idx_z0_im = floor(bg_coord_z);
bg_idx_x0_im = repmat(floor(bg_coord_x), [image_height 1]);
bg_idx_y0_im = repmat(floor(bg_coord_y), [1 image_width]);

% Compute dx, dy, dz images: each pixel is the fractional distance from
% the floored (integer) bilateral grid sample.
dx = repmat(bg_coord_x - bg_idx_x0, [image_height 1]);
dy = repmat(bg_coord_y - bg_idx_y0, [1 image_width]);
dz = bg_coord_z - bg_idx_z0_im;

% Each weight_{x}{y}{z} is an image (height x width).
% Each element (i,j) contributes to exactly 8 voxels.
weight_000 = columnize((1 - dx) .* (1 - dy) .* (1 - dz));
weight_100 = columnize((    dx) .* (1 - dy) .* (1 - dz));
weight_010 = columnize((1 - dx) .* (    dy) .* (1 - dz));
weight_110 = columnize((    dx) .* (    dy) .* (1 - dz));
weight_001 = columnize((1 - dx) .* (1 - dy) .* (    dz));
weight_101 = columnize((    dx) .* (1 - dy) .* (    dz));
weight_011 = columnize((1 - dx) .* (    dy) .* (    dz));
weight_111 = columnize((    dx) .* (    dy) .* (    dz));

st_i = (1:(8 * num_pixels))';
st_bg_xx = columnize(...
    [(bg_idx_x0_im(:) + 1) (bg_idx_x0_im(:) + 2) (bg_idx_x0_im(:) + 1) (bg_idx_x0_im(:) + 2) (bg_idx_x0_im(:) + 1) (bg_idx_x0_im(:) + 2) (bg_idx_x0_im(:) + 1) (bg_idx_x0_im(:) + 2)]');
st_bg_yy = columnize(...
    [(bg_idx_y0_im(:) + 1) (bg_idx_y0_im(:) + 1) (bg_idx_y0_im(:) + 2) (bg_idx_y0_im(:) + 2) (bg_idx_y0_im(:) + 1) (bg_idx_y0_im(:) + 1) (bg_idx_y0_im(:) + 2) (bg_idx_y0_im(:) + 2)]');
st_bg_zz = columnize(...
    [(bg_idx_z0_im(:) + 1) (bg_idx_z0_im(:) + 1) (bg_idx_z0_im(:) + 1) (bg_idx_z0_im(:) + 1) (bg_idx_z0_im(:) + 2) (bg_idx_z0_im(:) + 2) (bg_idx_z0_im(:) + 2) (bg_idx_z0_im(:) + 2)]');
st_bg_uu = i * ones(8 * num_pixels, 1);
st_bg_vv = j * ones(8 * num_pixels, 1);
st_s = ones(8 * num_pixels, 1);

% Prune rows of the triplet vectors (*not* of st) where where grid indices go
% out of bounds. We only prune certain elements. The number of rows of the
% output matrix is still the same.
indices = (st_bg_xx > 0) & (st_bg_xx <= grid_width) ...
    & (st_bg_yy > 0) & (st_bg_yy <= grid_height) ...
    & (st_bg_zz > 0) & (st_bg_zz <= grid_depth);

st_i = st_i(indices);
st_bg_xx = st_bg_xx(indices);
st_bg_yy = st_bg_yy(indices);
st_bg_zz = st_bg_zz(indices);
st_bg_uu = st_bg_uu(indices);
st_bg_vv = st_bg_vv(indices);
st_s = st_s(indices);
st_j = sub2ind(grid_size, st_bg_yy, st_bg_xx, st_bg_zz, st_bg_uu, st_bg_vv);
st_m = 8 * num_pixels;
st_n = num_grid_cells;

st = sparse(st_i, st_j, st_s, st_m, st_n);

w_i = columnize(repmat(1:num_pixels, [8 1]));
w_j = (1:(8 * num_pixels))';
w_s = columnize(...
    [weight_000, weight_100, weight_010, weight_110, weight_001, weight_101 weight_011, weight_111]');
w_m = num_pixels;
w_n = 8 * num_pixels;

w = sparse(w_i, w_j, w_s, w_m, w_n);
