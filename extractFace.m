function [cropped_Rhappy] = extractFace(image)
faceDetector = vision.CascadeObjectDetector();
bbox = step(faceDetector, image);
% boxFace = insertObjectAnnotation(Rhappy, 'rectangle', bbox, 'Face');

cropped_Rhappy = imcrop(image, bbox);
