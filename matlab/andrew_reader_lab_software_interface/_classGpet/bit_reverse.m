% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function ii = bit_reverse(objGpet, mm)
    nn = 2^ceil(log2(mm));
    ii = bin2dec(fliplr(dec2bin(0:(nn-1))));
    ii = ii(ii < mm);

end
