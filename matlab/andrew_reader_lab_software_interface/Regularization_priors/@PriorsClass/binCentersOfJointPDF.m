function [binCenter,nBins,binwidth] = binCentersOfJointPDF(ObjPrior,f,nBins)

miny = min(abs(f(:)));
maxy = max(abs(f(:)))+5;

nBins = min(nBins,floor(maxy));
binBoundery = floor(linspace(miny,maxy,nBins+1));

binwidth = binBoundery(2) - binBoundery(1);

binCenter = floor(binBoundery(1:end-1) + binwidth/2);
end