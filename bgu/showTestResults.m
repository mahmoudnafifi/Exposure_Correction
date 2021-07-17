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

% showTestResults(tr)
%
% Show the test results from a run of testBGU.
function [] = showTestResults(tr)

figure;
imshow(tr.input_fs);
title('Input full size');

figure;
imshow(tr.input_ds);
title('Input downsampled');

figure;
imshow(tr.output_ds);
title('Output downsampled');

figure;
imshow(tr.result_fs);
title('Result full size');

figure;
imshow(tr.output_fs);
title('Ground Truth full size');

pos = get(gcf, 'Position');

figure;
imshow(abs(tr.output_fs - tr.result_fs));
colorbar;
title('Absolute difference');
set(gcf, 'Position', pos);
