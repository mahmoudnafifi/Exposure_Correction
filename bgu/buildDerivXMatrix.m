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

function A = buildDerivXMatrix(grid_size)

% d/dx for every entry in the first slice except the last column.
m = grid_size(1) * (grid_size(2) - 1);
n = grid_size(1) * grid_size(2);
e = ones(m, 1);
d_dx = spdiags([-e, e], [0, grid_size(1)], m, n);

A = sparse(0, 0);

for v = 1:grid_size(5)
    for u = 1:grid_size(4)
        for k = 1:grid_size(3)
            A = blkdiag(A, d_dx);
        end
    end
end
