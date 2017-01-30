
function writeBinaryFile(dataFile,filename,format)
if nargin ==2
    format = 'float32';
end
fid = fopen(filename,'w');
if fid==-1
    error('failed to open %s\n',filename);
end

fwrite(fid,dataFile,format);
fclose(fid);
end