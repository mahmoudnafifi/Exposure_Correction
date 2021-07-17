% This is a 1-dimensional 5-tap low pass filter. It is used as a 2D separable low
% pass filter for constructing Gaussian and Laplacian pyramids.
%
% tom.mertens@gmail.com, August 2007
%

function f = pyramid_filter_
f = [.0625, .25, .375, .25, .0625];