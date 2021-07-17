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

function A = buildSecondDerivZMatrix(grid_size)

% d/dz for every entry in the cube except the first and last slice.
m = grid_size(1) * grid_size(2) * (grid_size(3) - 2);
n = grid_size(1) * grid_size(2) * grid_size(3);
e = ones(m, 1);
interior = spdiags([e, -2 * e, e], ...
    [0, grid_size(1) * grid_size(2), 2 * grid_size(1) * grid_size(2)], ...
    m, n);

boundary_z1 = makeBoundaryZ1(n);
boundary_zend = makeBoundaryZEnd(n);

% The matrix for a full cube.
cube = [boundary_z1; interior; boundary_zend];

% Repeat the cube for the last 2 dimensions.
A = sparse(0, 0);
for v = 1:grid_size(5)
    for u = 1:grid_size(4)
        A = blkdiag(A, cube);
    end
end

% Boundary conditions for the first slice.
% n is the number of columns for the full matrix (the number of variables
% in the grid).
function A = makeBoundaryZ1(n)
    mm = grid_size(1) * grid_size(2);
    nn = grid_size(1) * grid_size(2) * 2;
    e = ones(mm, 1);
    B = spdiags([-e, e], [0, grid_size(1) * grid_size(2)], mm, nn);
    % Concat zero columns to the right so we can vertical concat with the full
    % matrix later.
    A = [B, sparse(mm, n - nn)];
end

% Boundary conditions for the last slice.
function A = makeBoundaryZEnd(n)
    mm = grid_size(1) * grid_size(2);
    nn = grid_size(1) * grid_size(2) * 2;
    e = ones(mm, 1);
    B = spdiags([e, -e], [0, grid_size(1) * grid_size(2)], mm, nn);
    % Concat zero columns to the right so we can vertical concat with the full
    % matrix later.
    A = [sparse(mm, n - nn), B];
end


end
