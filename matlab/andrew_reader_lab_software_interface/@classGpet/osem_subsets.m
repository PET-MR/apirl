% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function  osem_subsets(objGpet, nsub,nAngles)
    if nsub==nAngles
        objGpet.sinogram_size.subsize = 1;
        objGpet.sinogram_size.subsets = 1:nAngles;
        s = subsets;
        st = 1 + bit_reverse(objGpet, nsub);
        for i= 1:nsub
            objGpet.sinogram_size.subsets(:,i) = s(:,st(i));
        end
        return
    end

    if rem(nAngles/nsub,2)~=0
        i = 1:nAngles/2;
        j = ~mod(nAngles/2./i,1);
        error(['Choose a valid subset: '  sprintf('%d ',i(j))])
    end

    objGpet.sinogram_size.subsize = nAngles /nsub;
    objGpet.sinogram_size.subsets = zeros(objGpet.sinogram_size.subsize, nsub);

    for j = 1:nsub
        k = 0;
        for i = j:nsub:nAngles/2
            k = k+1;
            objGpet.sinogram_size.subsets(k,j) = i;
            objGpet.sinogram_size.subsets(k+objGpet.sinogram_size.subsize/2,j) = i+nAngles/2;
        end
    end

    s = objGpet.sinogram_size.subsets;
    st = 1 + bit_reverse(objGpet, nsub);
    for i= 1:nsub
        objGpet.sinogram_size.subsets(:,i) = s(:,st(i));
    end
end