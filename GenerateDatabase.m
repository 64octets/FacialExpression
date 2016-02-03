% Generate database
% First we gonna collect pictures of faces: Happy, Sad, Neutral
% Use ROI to crop the images
% Generated manually. Such a pain.
clear;
close all;

ROI = 0;
CROP = 0;
UNIFY = 1;


%% step 1: ROI to manually select lips 

if(ROI)
face = imread('./database/Msad.jpg');
face = extractFace(face);

n=size(face,1)/2;
face=face(n+1:end,:,:);

BW = roipoly(face);
figure; imshow(BW);
return
end


%% step 2: convert logical file to double or uint8 and cropped the region.

if(CROP)
template = openfig('./database/s4.fig');

% to retrieve the handle to current figure
fig_h = get(gca,'children')

% to find the handle of the image
handleToImage2 = findobj(fig_h, 'Type', 'Image')

% reading the matrix data
template=get(handleToImage2,'CData' )
% template = im2double(template);

[rows,cols ]= size(template);
top = 0; bottom = 0; right = 0; left = 0;
% top
for r = 1:rows
    for c= 1:cols
        if( template(r,c) == 1 )
            top = r;
            break;
        end
    end
    if (top ~= 0) 
        break;
    end
end
% bottom
for r = rows:-1:1
    for c= 1:cols
        if( template(r,c) == 1 )
            bottom = r;
            break;
        end
    end
    if (bottom ~= 0) 
        break;
    end
end
disp(size(template));
disp(top);
disp(bottom);
width = cols;
height = bottom - top;
disp(height);
figure,
template = imcrop(template, [ 1 top width height]);

% recalculate the rows 
[rows,cols ]= size(template);
% right
for c = 1:cols
    for r= 1:rows
        if( template(r,c) == 1 )
            left = c;
            break;
        end
    end
    if (left ~= 0) 
        break;
    end
end
% bottom
for c = cols:-1:1
    for r= 1:rows
        if( template(r,c) == 1 )
            right = c;
            break;
        end
    end
    if (right ~= 0) 
        break;
    end
end
width = right - left;
height = rows;
template = imcrop(template, [ left 1 width height]);
figure,
imshow(template);



imwrite(template, './database/lips/2/4.jpg');

return
end

%% find the biggest dimensions of a feature, and unify the dimensions of that feature.
if(UNIFY)
    unifySize('lips');
end


return
