function [Features] = Eye(Face)

EyeDetector = vision.CascadeObjectDetector('EyePairBig');
Eyebox = step(EyeDetector, Face);
% box [x y width height]
features_position = Eyebox;
features_label = 'Eyes';
% imshow(insertObjectAnnotation(Face,'rectangle', features_position, features_label));

if(size(Eyebox,1) > 1)
    disp('more than one pair of eyes detected');
%   mouth lives in the lower half of the image.
%   find the widest width
    index= Eyebox(:,3)==max(Eyebox(:,3));
    Eyebox = Eyebox(index,:);
    
end
Features = imcrop(Face,Eyebox);
% insertObjectAnnotation(Face,'rectangle', Eyebox, features_label);

% figure;
%subplot(1,2,1);imshow(happyFeatures);title('Happy');
%subplot(1,2,2);
% imshow(sadFeatures); %title('Sad');
