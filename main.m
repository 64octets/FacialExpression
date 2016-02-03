%% Load and display images. Happy and sad. Extract Hue, saturation, and value. 11/19/15 by Sunnie
clear;
close all;
happy = imread('./database/annie.jpg');
% sad = imread('Rsad.jpg');


%% [Geometric] Extract face portion. Locate eye.
happyFace = extractFace(happy);
imwrite(happyFace,'./faces/1.jpg');

%% [Geometric] getEye template
template_eye = Eye(happyFace);
% figure('Name', 'Eye template'), imshow(template_eye);
gray = rgb2gray(template_eye);
template_eye = edge(gray,'Canny');
% figure, imshow(gray);
imwrite(template_eye,'./eyes/1.jpg');


%% [Geometric] get mouth and Eye brow template
template_mouth = Mouth(happyFace);
gray = rgb2gray(template_mouth);
template_mouth = edge(gray,'Canny');
% figure, imshow(gray);

%% FIS %%
%% Double precision Gray image
gray_face = im2double(rgb2gray(Mouth(happyFace)));
% gray_face = rgb2gray(template_mouth);
double_gray = im2double(gray);
Gx = [-1 1];
Gy = Gx';
Ix = conv2(gray_face,Gx,'same');
Iy = conv2(gray_face,Gy,'same');
% %% Obtain Gradient
% figure; image(Ix,'CDataMapping','direct'); colormap('gray'); title('Ix');
% figure; image(Iy,'CDataMapping','direct'); colormap('gray'); title('Iy');
% % figure('Name', 'Mouth template'), imshow(template_eye);
imwrite(template_mouth,'./mouths/1.jpg')
%% Fuzzy 
edgeFIS = newfis('edgeDetection');
edgeFIS = addvar(edgeFIS,'input','Ix',[-1 1]);
edgeFIS = addvar(edgeFIS,'input','Iy',[-1 1]);
sx = 0.05; sy = 0.05;
edgeFIS = addmf(edgeFIS,'input',1,'zero','gaussmf',[sx 0]);
edgeFIS = addmf(edgeFIS,'input',2,'zero','gaussmf',[sy 0]);
edgeFIS = addvar(edgeFIS,'output','Iout',[0 1]);
% start, end, peak (white/ black)
wa = 0; wb = 1; wc = 1;
ba = 0; bb = 0; bc = 1;
edgeFIS = addmf(edgeFIS,'output',1,'white','trimf',[wa wb wc]);
edgeFIS = addmf(edgeFIS,'output',1,'black','trimf',[ba bb bc]);
% figure
% subplot(2,2,1); plotmf(edgeFIS,'input',1); title('Ix');
% subplot(2,2,2); plotmf(edgeFIS,'input',2); title('Iy');
% subplot(2,2,[3 4]); plotmf(edgeFIS,'output',1); title('Iout')

%% FIS Rules
r1 = 'If Ix is zero and Iy is zero then Iout is white';
r2 = 'If Ix is not zero or Iy is not zero then Iout is black';
r = char(r1,r2);
edgeFIS = parsrule(edgeFIS,r);
showrule(edgeFIS)
%% Evaluate FIS
Ieval = zeros(size(gray_face));% Preallocate the output matrix
for ii = 1:size(gray_face,1)
    Ieval(ii,:) = evalfis([(Ix(ii,:));(Iy(ii,:));]',edgeFIS);
end
% for col = 1:size(gray_face,1)
%     for row = 1:size(gray_face,2)
%         if (Ieval(col,row) < 0.6)
%             Ieval(col,row) = 0;
%         end
%     end
% end
%% plot result
% figure; image(gray_face,'CDataMapping','scaled'); colormap('gray');
% title('Original Grayscale Image')
% 
% figure; image(Ieval,'CDataMapping','scaled'); colormap('gray');
% title('Edge Detection Using Fuzzy Logic')

gauss_face = imgaussfilt(Ieval);
figure;imshow(gauss_face);title('GaussFilter')
% return
%% imfill

Morph = edgeFill(gray);
figure;imshow(Morph);title('morph')
% imwrite(result,'./mouths/2.jpg')
% clean the edge of the picture...

% figure; imshow(Morph);

return
BW2= Morph; %= imdilate(Canny,SE);
% BW2= imfill(BW2,'holes');
% BW2= imerode(BW2, SE);
n=size(BW2,1)/2;
BW2=BW2(n+1:end,:,:);
BW2= imfill(BW2,'holes');

figure;imshow(BW2);title('imFill')
BW2= imdilate(BW2,SE);
BW2= imfill(BW2,'holes');
BW2= imerode(BW2, SE);
figure;imshow(BW2);title('imFill')
% BW2= imfill(BW2,'holes');
return

% %% FCM for image
% img = double(gray); %rgb2gray(imread('Rhappy.jpg')));
% 
% clusterNum = 3;
% [ Unow, center, now_obj_fcn ] = FCMfowrImage( img, clusterNum );
% figure;
% subplot(2,2,1); imshow(img,[]);
% for i=1:clusterNum
%     subplot(2,2,i+1);
%     imshow(Unow(:,:,i),[]);
% end

% %% [Gabor filter]
% wavelength = 5;
% orientation = 90;
% [mag,phase] = imgaborfilt(Ieval,wavelength,orientation);
% figure
% subplot(1,3,1);
% imshow(Ieval);
% title('Original Image');
% subplot(1,3,2);
% imshow(mag,[])
% title('Gabor magnitude');
% subplot(1,3,3);
% imshow(phase,[]);
% title('Gabor phase');
% return
%% Erosion and Dilation
SE = strel('diamond',2);
happyFace = imdilate(happyFace,SE);
happyFace = imerode(happyFace, SE);

% Convert HSV
happyFace_hsv = rgb2hsv(happyFace);
doubleHSV = double(happyFace_hsv);
[rows,cols, three] = size(happyFace_hsv);
BW = zeros(rows,cols);
H0 = 0.06;
S0 = 0.05;
for col = 1:cols
    for row = 1:rows
        if ( doubleHSV(row,col,1) > H0 && doubleHSV(row,col,2) > S0)
           BW(row,col) = 1;
        else
           BW(row,col) = 0;
        end
    end
end
figure;
imshow(BW);title('HSV filter')
return
%% [Color-Based] Visit every pixel in of the image the turn unskin area black.
[rows,cols, three] = size(happyFace_hsv);
% Note: at this point, pixels of happyFace are stored as uint8 and scaled
% [0, 255], yet I still mulitplied the values by 255 and it worked. SO
% WEIRD.

for col = 1:cols
    for row = 1:rows
        R = happyFace(row,col,1) * 255;     G = happyFace(row,col,2) * 255;     B = happyFace(row,col,3) * 255;
        H = happyFace_hsv(row,col,1) * 255; S = happyFace_hsv(row,col,2) * 255; V = happyFace_hsv(row,col,3) * 255;
        if not ( (( (0.836*G - 14) < B ) && ((0.86*G + 44) > B)) || ((0.79*G - 67 < B) && ((0.78 * G + 42) > B)))
            if( H> 19) && (H <240)
                happyFace(row,col,1) = 0;
                happyFace(row,col,2) = 0;
                happyFace(row,col,3) = 0;
            end    
        end
    end
end

happyFace = imdilate(happyFace,SE);
happyFace = imerode(happyFace, SE);
% figure, imshow(happyFace);

gray = rgb2gray(happyFace);
gauss_face = imgaussfilt(gray);
gauss_face = imgaussfilt(gauss_face,4);
% bw_face = im2bw(gauss_face,0.3);
% figure, imshow(bw_face);
figure,imshow(imdilate(edge(gray,'Canny'), SE));
% [gx, gy] = gradient(happyFace);
[gx, gy] = imgradient(gray);
figure, imshow(gx,[]);
figure, imshow(gy,[]);
% imshow(edge(gray,'Canny'));
% up to this point, everything is fine and done.

%% store the skin pixels to skin.bmp
imwrite(happyFace,'skin.bmp');
 

%% [Color-Based] [Not effective] New Method: use Hue to detect lip

% cut off the upper half. Mouth is not there.
n=size(happyFace,1)/2;
halfFace=happyFace(n+1:end,:,:);
n=size(halfFace,2)/6 ;
n2=size(halfFace,2)/6 * 5;
halfFace=halfFace(:,n:n2,:);
[rows,cols, three] = size(halfFace);

% Convert color code to HSV and YCBCr
happyFace_hsv = rgb2hsv(halfFace);
happyFace_ycbcr = rgb2ycbcr(halfFace);

figure('Name', 'half'), imshow(halfFace);
figure('Name', 'Hue'), imshow(happyFace_hsv(:,:,1));
figure('Name', 'Saturation'), imshow(happyFace_hsv(:,:,2));
figure('Name', 'YCbCr'), imshow(happyFace_ycbcr);
figure('Name', 'Cr'), imshow(happyFace_ycbcr(:,:,3));

Cr2=0;
Cr_Cb=0;
for row = 1:rows
    for col = 1:cols
        Cr2 = Cr2 + double(normalize(happyFace_ycbcr(row,col,3).^2,256));
        Cr_Cb = Cr_Cb + double(normalize(happyFace_ycbcr(row,col,3),256)/normalize(happyFace_ycbcr(row,col,2),256));
        
    end
end
mew = 0.95 * (Cr2/Cr_Cb);

lipMap = zeros(rows,cols);
for row = 1:rows
    for col = 1:cols
        CrSquare = double(normalize(happyFace_ycbcr(row,col,3).^2,256));
        ratio = normalize(double(normalize(happyFace_ycbcr(row,col,3),256)/normalize(happyFace_ycbcr(row,col,2),256)),256);
        lipMap(row,col) = happyFace_hsv(row,col,2) * CrSquare * (CrSquare - mew * ratio).^2;     
%         disp(lipMap(row,col));
    end
end
figure('Name', 'lipmap'), imshow(lipMap);

%% Normalize the luminance level in the image.
% let the pixels be stored as doubles and in range [0,1]
happyFace_double = im2double(halfFace);

% normalize the image (make it grayish (lip stands out)
normalizedFace = normalizeTransform(happyFace_double);
% figure('Name','Normalized'),imshow(normalizedFace);

%% Apply DHT (RGB ==> C1C2C3). use color (high in red) to locate lip.

% from the tutorial, it uses the below matrix to transfrom RGB channels to
% C1(luminance), C2, C3 (chrominance)
DHT_matrix = [0.5773 0.5773 0.5773; 0.5773 0.2113 -0.7886;0.5773 -0.7886 0.2113];
[rows, cols, three] = size(halfFace);
if ( 1 )
    for col = 1:cols
        for row = 1:rows
          mult = [double(halfFace(row,col,1))/255; double(halfFace(row,col,2))/255; double(halfFace(row,col,3))/255];
%             mult = [double(normalizedFace(row,col,1)); double(normalizedFace(row,col,2)); double(normalizedFace(row,col,3))];
            
%             disp(mult);
%           disp(size(mult));
            tmp = double(DHT_matrix) * (mult); 
            normalizedFace(row,col,1) = (tmp(1));
            normalizedFace(row,col,2) = (tmp(2));
            normalizedFace(row,col,3) = (tmp(3));
        end
    end
%   yeah, the third one does has a slighty stand-out lip. 




% figure('Name','DHT Applied'),imshow(normalizedFace(:,:,3));
end

%% Gaussian filter
gaussianFace = imgaussfilt(normalizedFace(:,:,3));


gaussianFace= imdilate(gaussianFace,SE);
gaussianFace = imerode(gaussianFace, SE);

figure('Name','Gaussian'),imshow(gaussianFace);
C3max = 0;
for col = 1:cols
    for row = 1:rows
        if(gaussianFace(row,col) > C3max)
            C3max = gaussianFace(row,col);
        end
    end
end

thres = C3max/4;
disp(C3max);
for col = 1:cols
    for row = 1:rows
        if( gaussianFace(row,col) <=  thres|| (happyFace_hsv(row,col,1) < 0.5&&(happyFace_hsv(row,col,2) < 0.5)))
            gaussianFace(row,col) = 0;
        else
            gaussianFace(row,col) = 1;
        end
    end
end
figure('Name', 'Threshold'), imshow(gaussianFace);
%  FFT_face = fft(happyFace);






%%
% labelFeatures(sadFace, 'Sad');
% labelFeatures(dil, 'Happy');
% rectangle('Position',Mouthbox,'LineWidth',4,'LineStyle','-','EdgeColor','g');
% title('Mouth');


% negativeImage_mouth = imread('negative_mouth.jpg');
% negativeFolder = fullfile('/media/storage/FacialExpression');
% load('positiveInstances_Rhappy_mouth.mat');
% % % trainCascadeObjectdetector('mouth.xml',positiveInstances_Rhappy_mouth,negativeFolder);
% MouthDetector = vision.CascadeObjectDetector('mouth','MergeThreshold',32);
% Mouthbox = step(MouthDetector, happyFace);
% features_position = Mouthbox;%[Mouthbox; Eyebox];
% Mouthbox = step(MouthDetector, sadFace);

% features_label = 'Mouth';%['Mouth' 'Eyes'];

