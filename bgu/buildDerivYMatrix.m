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

function A = buildDerivYMatrix(grid_size)

% Derivatives down y.
ny = grid_size(1);
e = ones(ny - 1, 1);
d_dy = spdiags([-e, e], 0:1, ny - 1, ny);

A = sparse(0, 0);

for v = 1:grid_size(5)
    for u = 1:grid_size(4)
        for k = 1:grid_size(3)
            for j = 1:grid_size(2)
                A = blkdiag(A, d_dy);
            end
        end
    end
end
