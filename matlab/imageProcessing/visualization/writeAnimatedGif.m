% Function that writes an animated gif from a 3d volume:
function filename = writeAnimatedGif(volume, filename)

slices = find(mean(mean(volume))>0);
maxImage = max(max(max(volume)));
for n = 1 : numel(slices)
    imind = uint8(round(double(volume(:,:,slices(n))./maxImage.*255)));
    cm = repmat([0 : 1/255 : 1]',1,3);
    % Write to the GIF File 
    if n == 1 
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
      imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', 0.05); 
    end 
end