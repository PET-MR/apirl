
function out = ParseInterfileDataFile(directory,report)

% Parse the interfile data folder of SIEMSE mMR scanner
% output: number of list-mode data, sinograms, normalization files and
% their interfile headers plus the number of unclassified files


if nargin==1, report = 0; end

nInterfileCompressedSinogramFiles = 0;
nInterfileUncompressedSinogramFiles = 0;
nInterfileListModeFiles = 0;
nInterfileNormFiles = 0;
nInterfileHumanUmaps = 0;
nInterfileHardwareUmaps = 0;
%             Unclassified = 0;
[CompressedSinogramHdrs,UncompressedSinogramHdrs,ListModeHdrs, NormFileHdrs,HumanUmapHdrs, HardwareUmapHdrs] = deal(struct('hdr',[]));

listing = dir(directory);
for i = 3:length(listing)
    
    [~, name, ext] = fileparts(listing(i).name); % This name doesnt include the path.
    if ~isempty(name) && ~isempty(ext)
        if strcmpi(ext,'.l')
            nInterfileListModeFiles = nInterfileListModeFiles + 1;
            hdrFilename = [directory name '.l.hdr'];
            ListModeHdrs = getHeaderFiles(hdrFilename,ListModeHdrs,nInterfileListModeFiles);
        elseif strcmpi(ext,'.s')
            hdrFilename = [directory name '.s.hdr'];
            if ~isempty(strfind(name,'uncomp'))
                nInterfileUncompressedSinogramFiles = nInterfileUncompressedSinogramFiles + 1;
                UncompressedSinogramHdrs = getHeaderFiles(hdrFilename,UncompressedSinogramHdrs,nInterfileUncompressedSinogramFiles);
            else
                nInterfileCompressedSinogramFiles = nInterfileCompressedSinogramFiles + 1;
                CompressedSinogramHdrs = getHeaderFiles(hdrFilename,CompressedSinogramHdrs,nInterfileCompressedSinogramFiles);
            end
            
        elseif strcmpi(ext,'.n')
            hdrFilename = [directory name '.n.hdr'];
            nInterfileNormFiles = nInterfileNormFiles + 1;
            NormFileHdrs = getHeaderFiles(hdrFilename,NormFileHdrs,nInterfileNormFiles);
        elseif strcmpi(ext,'.v')
            hdrFilename = [directory name '.v.hdr'];
            if ~isempty(strfind(name,'hardware'))
                nInterfileHardwareUmaps = nInterfileHardwareUmaps + 1;
                HardwareUmapHdrs = getHeaderFiles(hdrFilename,HardwareUmapHdrs,nInterfileHardwareUmaps);
            end
            if ~isempty(strfind(name,'human'))
                nInterfileHumanUmaps = nInterfileHumanUmaps + 1;
                HumanUmapHdrs = getHeaderFiles(hdrFilename,HumanUmapHdrs,nInterfileHumanUmaps);
            end
        else
            
        end
    else
        %         Unclassified = Unclassified + 1;
    end
end

out.CompressedSinogramHdrs = CompressedSinogramHdrs;
out.nInterfileCompressedSinogramFiles = nInterfileCompressedSinogramFiles;

out.UncompressedSinogramHdrs = UncompressedSinogramHdrs;
out.nInterfileUncompressedSinogramFiles = nInterfileUncompressedSinogramFiles;

out.ListModeHdrs = ListModeHdrs;
out.nInterfileListModeFiles = nInterfileListModeFiles;

out.NormFileHdrs = NormFileHdrs;
out.nInterfileNormFiles = nInterfileNormFiles;

out.HumanUmapHdrs = HumanUmapHdrs;
out.nInterfileHumanUmaps = nInterfileHumanUmaps;

out.HardwareUmapHdrs = HardwareUmapHdrs;
out.nInterfileHardwareUmaps = nInterfileHardwareUmaps;




if report
    fprintf('InterFiles: %d Compressed Sinograms, %d UnCompressed Sinograms,%d ListModeFiles, %d NormFiles, %d HumanUmaps, %d HardwareUmaps\n',...
        nInterfileCompressedSinogramFiles, nInterfileUncompressedSinogramFiles,nInterfileListModeFiles, nInterfileNormFiles,...
        nInterfileHumanUmaps, nInterfileHardwareUmaps);
end
end

function dataStruct = getHeaderFiles(hdrFilename,dataStruct,n)
if exist(hdrFilename,'file')
    dataStruct(n).hdr = getInfoFromInterfile(hdrFilename);
else
    error('could not find the header file: %s\n',hdrFilename);
end
end