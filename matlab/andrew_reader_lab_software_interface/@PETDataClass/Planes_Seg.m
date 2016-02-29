function nPlanePerSeg = Planes_Seg (Gantry)


    a = (Gantry.MRD -(Gantry.Span+1)/2);
    b = floor(a/Gantry.Span)+ floor(Gantry.MRD/a);
    nseg = 2*b+1;

    a = ones(Gantry.nCrystalRings);
    minRingDiff = [0,(Gantry.Span+1)/2:Gantry.Span:Gantry.MRD];

    s = zeros(1,(nseg+1)/2);
    for j = 1:(nseg+1)/2
        s(j) = length(diag(a,minRingDiff(j)));
    end

    if Gantry.Span>1, s = 2*s-1; end

    nPlanePerSeg = zeros(1,nseg);
    nPlanePerSeg(1) = s(1);
    nPlanePerSeg(2:2:end) = s(2:end);
    nPlanePerSeg(3:2:end) = s(2:end);

end