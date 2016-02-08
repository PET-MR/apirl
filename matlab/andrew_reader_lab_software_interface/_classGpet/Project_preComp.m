% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function lambda = Project_preComp(objGpet,X,g,Angles,RadialBins,dir)
         
 if size(X,3)==1 % 2D projection
     nAxialSymPlanes = 1;
     nonSymPlanes = {1};
     g.iSlices = 1;
     g.nPlanes = 1;
     g.AxialSym_pmt = [0 0 0];
 else
     nAxialSymPlanes = length(g.AxialSymPlanes);
     nonSymPlanes = cell(nAxialSymPlanes,1);
     for i = 1:nAxialSymPlanes
         nonSymPlanes{i} = find(g.AxialSym_pmt(:,1) == i);
     end
 end

 if dir ==-1 % back-projection
     lambda = zeros(g.iRows*g.iColumns*g.iSlices,1,'single');
     X(isnan(X))=0;
 elseif dir==1 % forward-projection
     lambda = zeros(g.nRad,g.nAng,g.nPlanes,'single');
 end

 for I = 1: nAxialSymPlanes
     P = objGpet.Geom{I}; % load the part of the system matrix covering Ith group of sino planes

     for p_idx = 1:length(nonSymPlanes{I}) %loop over planes with symmetric system matrix
         plane = nonSymPlanes{I}(p_idx);
         p = g.AxialSym_pmt(plane,:);

         for a_idx = 1:length(Angles)/2 % loop over angular bins
             ang = Angles(a_idx);
             symAng = ang + g.nAng/2;

             for b_idx = 1:length(RadialBins) % loop over radial bins
                 bin = RadialBins(b_idx);

                 M = double(P{ang,bin});
                 temp = (g.iRows*g.iColumns* (p(2)*M(:,3)+ p(3)));
                 ind1 = M(:,1)+1 + g.iRows*(g.iRows-1-M(:,2))+ temp;
                 ind2 = M(:,2)+1 + g.iRows*(M(:,1))+ temp;
                 G = M(:,4)/1e4;

                 if dir==-1 % back-projection
                     lambda(ind1) = lambda(ind1) + G*X(bin,ang,plane);%
                     lambda(ind2) = lambda(ind2) + G*X(bin,symAng,plane);%
                 elseif dir==1 % forward-projection
                     lambda(bin,ang,plane) = G'*X(ind1);%
                     lambda(bin,symAng,plane) = G'*X(ind2);%
                 end

             end
         end
     end
 end

 if dir ==-1 % back-projection
     lambda = reshape(lambda, g.iRows,g.iColumns,g.iSlices);
 end

end