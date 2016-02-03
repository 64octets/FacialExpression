% testing

target = imread('./mouth/h2.jpg');
figure,
imshow(target);
disp(emotionDetection(target,'lips'));
