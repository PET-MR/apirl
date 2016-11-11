function plot_histogram(ObjPrior,f,M)

maxy = max(f(:))*1.2;
[binCenter,nBins,~,binBoundery,fB]  = ObjPrior.binCentersOfJointPDF(f,M,maxy,1);


Hist = zeros(1,nBins);
    
    for ic = 1:nBins-1
        
        I = binBoundery(ic)<= fB &  fB < binBoundery(ic+1);
        Hist(ic) = sum(I(:));
        
    end


figure,
plot(binCenter,Hist)