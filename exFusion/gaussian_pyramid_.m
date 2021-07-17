% Contruction of Gaussian pyramid
%
% Arguments:
%   image 'I'
%   'nlev', number of levels in the pyramid (optional)
%
% tom.mertens@gmail.com, August 2007
%

function pyr = gaussian_pyramid_(I,nlev)

r = size(I,1);
c = size(I,2);

if ~exist('nlev')
    %compute the highest possible pyramid
    nlev = floor(log(min(r,c)) / log(2));
end

% start by copying the image to the finest level
pyr = cell(nlev,1);
pyr{1} = I;

% recursively downsample the image
filter = pyramid_filter_;
for l = 2:nlev
    I = downsample_(I,filter);
    pyr{l} = I;
end