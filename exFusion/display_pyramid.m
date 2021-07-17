% Display the contents of a pyramid, as returned by functions
% 'laplacian_pyramid' or 'gaussian pyramid'
%
% tom.mertens@gmail.com, August 2007
%

function display_pyramid(pyr)

L = length(pyr);
r = size(pyr{1},1);
c = size(pyr{1},2);
k = size(pyr{1},3);
R = zeros(r,2*c,k);

offset = 1;
for l = 1:L
    I = pyr{l};
    r = size(I,1);
    c = size(I,2);
    R(1:r, offset:offset-1+c, :) = I;
    offset = offset + c;
end

if (min(R(:)) < 1e-5)
    %make negative values displayable
    a = min(R(:));
    b = max(R(:));
    R = (R - a) / (b - a);
end    

figure; imshow(R);
