function [expression] = emotionDetection(target, featureName)
% emotion detection
% for each feature,
%   load the database 
%   use cross correlation to find the closest matched template
%   output the matched template, and final result (emotion)

%% lips matching


% load database for emotion 1
% database size
nITEMS = 4;
nEMOTIONS = 2;
maxOfsum = 0;
emotionIndex = -1;
matching = 0;
figure,
% for emotion = 1:nEMOTIONS
for emotion = 1:nEMOTIONS
    for k = 1:nITEMS
        % Create an image filename, and read it in to a variable called imageData.
        jpgFileName = strcat('./database/', featureName, '/',num2str(emotion), '/', num2str(k), '.jpg');
        if exist(jpgFileName, 'file')
            template = imread(jpgFileName);
            imshow(template);
%             pause(1);
            D = xcorr2(template, target);
            sum_ = max(D(:));
            if(maxOfsum < sum_)
                maxOfsum = sum_;
                emotionIndex = emotion;
                matching = template;
            end
        else
            fprintf('File %s does not exist.\n', jpgFileName);	
        end
    end
end
if (emotionIndex == 1)
    expression = 'happy';
elseif (emotionIndex == 2)
    expression = 'sad';
else
    expression = 'neutral';
end

imshow(matching),title(expression);


end
        
