% GOALS: unify the dimensions of the templates
% find the biggest width
% pad the other images so all have the same width
% find the biggest length
% pad the other images so all have the same height

function unifySize(featureName)
% database size
nITEMS = 4;
nEMOTIONS = 2;


%% find max dimensions [maxwidth, maxheight]

maxWidth = -1;
maxHeight = -1;
for emotion = 1:nEMOTIONS
for k = 1:nITEMS
% Create an image filename, and read it in to a variable called imageData.
jpgFileName = strcat('./database/', featureName, '/',num2str(emotion), '/', num2str(k), '.jpg');
if exist(jpgFileName, 'file')
    imageData = imread(jpgFileName);	
    if maxWidth < size(imageData, 1)
        maxWidth = size(imageData, 1);
    end
    if maxHeight < size(imageData, 2)
        maxHeight = size(imageData, 2);
    end
else
    fprintf('File %s does not exist.\n', jpgFileName);	
end
end
end 

%% padding
for emotion = 1:nEMOTIONS
for k = 1:nITEMS
% Create an image filename, and read it in to a variable called imageData.
jpgFileName = strcat('./database/', featureName, '/',num2str(emotion), '/', num2str(k), '.jpg');
if exist(jpgFileName, 'file')
    imageData = imread(jpgFileName);	
    if (maxWidth > size(imageData, 1)) || (maxHeight > size(imageData, 2))
        himpad = vision.ImagePadder('SizeMethod', 'Output size','NumOutputColumnsSource','Property','NumOutputColumns', maxHeight,'NumOutputRowsSource','Property','NumOutputRows', maxWidth);
        temp = step(himpad, imageData);
        figure,
        imshow(temp);
        imwrite(temp, jpgFileName);
    end
else
    fprintf('File %s does not exist.\n', jpgFileName);	
end
end
end 

end
