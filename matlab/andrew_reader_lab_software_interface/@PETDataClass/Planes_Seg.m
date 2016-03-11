function nPlanePerSeg = Planes_Seg (PETData)

MRD = PETData.Gantry.maxRingDiff;
Span = PETData.span;
nCrystalRings = PETData.Gantry.nCrystalRings;

a = (MRD -(Span+1)/2);
b = floor(a/Span)+ floor(MRD/a);
nseg = 2*b+1;

a = ones(nCrystalRings);
minRingDiff = [0,(Span+1)/2:Span:MRD];

s = zeros(1,(nseg+1)/2);
for j = 1:(nseg+1)/2
    s(j) = length(diag(a,minRingDiff(j)));
end

if Span>1, s = 2*s-1; end

nPlanePerSeg = zeros(1,nseg);
nPlanePerSeg(1) = s(1);
nPlanePerSeg(2:2:end) = s(2:end);
nPlanePerSeg(3:2:end) = s(2:end);

end