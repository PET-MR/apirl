
function overlapImages(imageBackground, imageOverlay, alpha, mapaColores)
h = figure;
imshow((imageBackground-min(min(imageBackground)))/(max(max(imageBackground))-min(min(imageBackground))));
hold on
imO = imshow(imageOverlay./max(max(imageOverlay)));
imAlphaData = alpha;
set(imO,'AlphaData',imAlphaData);
colormap(mapaColores);