


%% Load and display images. Happy and sad. Extract Hue, saturation, and value. 11/19/15 by Sunnie
clear;
close all;
happy = imread('./database/ksad.jpg');
% sad = imread('Rsad.jpg');

% %% Define constants
% LIP = 1;
% EYEBROW = 2;

%% [Geometric] Extract and save face portion.
happyFace = extractFace(happy);
imwrite(happyFace,'./faces/5.jpg');

%% [Geometric] getEye template
template_eye = Eye(happyFace);
% figure('Name', 'Eye template'), imshow(template_eye);
gray = rgb2gray(template_eye);
template_eye = edge(gray,'Canny');
% figure, imshow(gray);
imwrite(template_eye,'./eyes/s5.jpg');

%% [Geometric] get mouth and Eye brow template
template_mouth = Mouth(happyFace);
gray_mouth = rgb2gray(template_mouth);
template_mouth = edge(gray_mouth,'Canny');
figure, imshow(gray_mouth);

%% imfill
Morph = edgeFill(gray_mouth);

SE = strel('diamond',1);
Morph = imerode(Morph, SE); 
Morph = imdilate(Morph, SE); 
figure;imshow(Morph);title('morph')
% imwrite(Morph, './mouth/s5.jpg');
Morph = im2double(Morph);
disp(emotionDetection(Morph,'lips'));

