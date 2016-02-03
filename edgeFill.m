function [ result ] = edgeFill( gray )
%EDGEFILL Summary of this function goes here
%   Detailed explanation goes here

SE = strel('diamond',1);
% gray = rgb2gray(happyFace);
% figure;imshow(edge(gray,'Canny'));title('Canny')
h = fspecial('sobel');
smoothed = imfilter(gray,h);
figure;imshow(smoothed);title('imfilter')

Canny = edge(smoothed,'Canny');
% Canny = edge(gray, 'Canny');
figure;imshow(Canny);title('From EdgeFill function')
Morph = Canny;

Morph= imfill(Morph,'holes');
Morph = bwmorph(Morph,'remove');
% Morph = bwmorph(Morph,'thicken');
Morph = bwmorph(Morph,'bridge');
Morph= imfill(Morph,'holes');

height = size(Morph,1);
width = size(Morph, 2);
xmin = uint8(width * 0.1);
ymin = uint8(height * 0.1);

width = uint8(width - xmin -xmin);
height = uint8(height - ymin -ymin);

result = imcrop(Morph, [xmin ymin width height]);



% Morph = imcrop(Morph, 
% Morph = bwmorph(Morph,'bridge');
% Morph = imdilate(Morph, SE);
% Morph = imerode(Morph, SE);
% Morph = bwmorph(Morph,'thicken');
% Morph = imerode(Morph, SE);
% Morph = bwmorph(Morph,'spur');
% Morph = bwmorph(Morph,'remove');
% Morph= imfill(Morph,'holes');
end
