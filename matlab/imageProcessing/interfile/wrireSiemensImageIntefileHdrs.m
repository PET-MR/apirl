
function wrireSiemensImageIntefileHdrs(ImageInterFileName,SinoInfo,NormInfo,ImgInfo)

% SinoInfo: interfile header of sinogram data or a structure contaning the header info
% NormInfo: interfile header of sinogram data or a structure contaning the header info
% ImgInfo: a struct info of the image with the following fields
    % ImgInfo.matrixSize = [ , , ];
    % ImgInfo.voxelSize_mm = [ , , ];
    % ImgInfo.nIter = ;
    % ImgInfo.nSubsets = ;
    % ImgInfo.ReconMethod = '';
    % ImgInfo.MinMaxImgCount = [ , ];
    %ImgInfo.scaleFactor

if ischar(SinoInfo) 
    if ~exist(SinoInfo,'file'), error('could not find: %s\n',SinoInfo); end
    SinoInfo = getInfoFromSiemensIntf(SinoInfo);
end
if ischar(NormInfo)
    if ~exist(NormInfo,'file'), error('could not find: %s\n',NormInfo); end
    NormInfo = getInfoFromSiemensIntf(NormInfo);
end
[~,dataFileName,ext]=fileparts(ImageInterFileName);
dataFileName = [dataFileName ext];
fid = fopen([ImageInterFileName '.hdr'],'w');
fprintf(fid,'!INTERFILE:=\n');
fprintf(fid,'%%comment:=created with code from Nov  5 2013 07:20:45\n');
fprintf(fid,'!originating system:=2008\n');
fprintf(fid,'%%SMS-MI header name space:=image subheader\n');
fprintf(fid,'%%SMS-MI version number:=3.4\n');
fprintf(fid,'\n');
fprintf(fid,'!GENERAL DATA:=\n');
fprintf(fid,'%%sinogram header file:=%s\n',[SinoInfo.NameOfDataFile '.hdr']);
fprintf(fid,'%%sinogram data file:=%s\n',SinoInfo.NameOfDataFile);
fprintf(fid,'!name of data file:=%s\n',dataFileName);
fprintf(fid,'\n');
fprintf(fid,'!GENERAL IMAGE DATA:=\n');
fprintf(fid,'%%study date (yyyy:mm:dd):=%s\n',SinoInfo.StudyDateYyyyMmDd);
fprintf(fid,'%%study time (hh:mm:ss GMT+00:00):=%s\n',SinoInfo.StudyTimeHhMmSsGmt0000);
fprintf(fid,'isotope name:=%s\n',SinoInfo.IsotopeName);
fprintf(fid,'isotope gamma halflife (sec):=%f\n',SinoInfo.IsotopeGammaHalflifeSec);
fprintf(fid,'isotope branching factor:=%f\n',SinoInfo.IsotopeBranchingFactor);
fprintf(fid,'radiopharmaceutical:=%s\n',SinoInfo.Radiopharmaceutical);
fprintf(fid,'%%tracer injection date (yyyy:mm:dd):=%s\n',SinoInfo.TracerInjectionDateYyyyMmDd);
fprintf(fid,'%%tracer injection time (hh:mm:ss GMT+00:00):=%s\n',SinoInfo.TracerInjectionTimeHhMmSsGmt0000);
fprintf(fid,'tracer activity at time of injection (Bq):=%e\n',SinoInfo.TracerActivityAtTimeOfInjectionBq);
fprintf(fid,'relative time of tracer injection (sec):=%f\n',SinoInfo.IsotopeBranchingFactor);
fprintf(fid,'injected volume (ml):=%f\n',SinoInfo.InjectedVolumeMl);
fprintf(fid,'image data byte order:=%s\n',SinoInfo.ImageDataByteOrder);
fprintf(fid,'%%patient orientation:=%s\n',SinoInfo.PatientOrientation);
fprintf(fid,'%%image orientation:={1,0,0,0,1,0}\n');
fprintf(fid,'!PET data type:=image\n');
fprintf(fid,'number format:=float\n');
fprintf(fid,'!number of bytes per pixel:=4\n');
fprintf(fid,'number of dimensions:=3\n');
fprintf(fid,'matrix axis label[1]:=x\n');
fprintf(fid,'matrix axis label[2]:=y\n');
fprintf(fid,'matrix axis label[3]:=z\n');
fprintf(fid,'matrix size[1]:=%f\n',ImgInfo.matrixSize(1));
fprintf(fid,'matrix size[2]:=%f\n',ImgInfo.matrixSize(2));
fprintf(fid,'matrix size[3]:=%f\n',ImgInfo.matrixSize(3));
fprintf(fid,'scale factor (mm/pixel) [1]:=%0.4f\n',ImgInfo.voxelSize_mm(1));
fprintf(fid,'scale factor (mm/pixel) [2]:=%0.4f\n',ImgInfo.voxelSize_mm(2));
fprintf(fid,'scale factor (mm/pixel) [3]:=%0.4f\n',ImgInfo.voxelSize_mm(3));
fprintf(fid,'horizontal bed translation:=%s\n', SinoInfo.HorizontalBedTranslation);
fprintf(fid,'start horizontal bed position (mm):=%0.4f\n',SinoInfo.StartHorizontalBedPositionMm);
fprintf(fid,'end horizontal bed position (mm):=%f\n',-((ImgInfo.matrixSize(3)-1)*ImgInfo.voxelSize_mm(3)) + SinoInfo.StartHorizontalBedPositionMm);
fprintf(fid,'start vertical bed position (mm):=%f\n',SinoInfo.StartVerticalBedPositionMm);
fprintf(fid,'%%reconstruction diameter (mm):=584.178\n');
fprintf(fid,'quantification units:=Bq/ml\n');
fprintf(fid,'%%scanner quantification factor (Bq*s/ECAT counts):=%e\n',NormInfo.ScannerQuantificationFactorBqSEcatCounts);
fprintf(fid,'%%decay correction:=reftime\n');
fprintf(fid,'%%decay correction reference date (yyyy:mm:dd):=%s\n',SinoInfo.StudyDateYyyyMmDd);
fprintf(fid,'%%decay correction reference time (hh:mm:ss GMT+00:00):=%s\n',SinoInfo.StudyTimeHhMmSsGmt0000);
fprintf(fid,'slice orientation:=transverse\n');
fprintf(fid,'method of reconstruction:=%s\n',ImgInfo.ReconMethod);
fprintf(fid,'%%gantry offset (mm) [1]:=0.0\n');
fprintf(fid,'%%gantry offset (mm) [2]:=0.0\n');
fprintf(fid,'%%gantry offset (mm) [3]:=0.0\n');
fprintf(fid,'%%gantry offset pitch (degrees):=0.0\n');
fprintf(fid,'%%gantry offset yaw (degrees):=0.0\n');
fprintf(fid,'%%gantry offset roll (degrees):=0.0\n');
fprintf(fid,'number of iterations:=%d\n',ImgInfo.nIter);
fprintf(fid,'%%number of subsets:=%d\n',ImgInfo.nSubsets);
fprintf(fid,'filter name:=ALL_PASS\n');
fprintf(fid,'%%xy-filter (mm):=0.0\n');
fprintf(fid,'%%z-filter (mm):=0.0\n');
fprintf(fid,'%%filter order:=0\n');
fprintf(fid,'%%image zoom:=1\n');
fprintf(fid,'%%x-offset (mm):=0.0\n');
fprintf(fid,'%%y-offset (mm):=0.0\n');
fprintf(fid,'applied corrections:={normalization,deadtime,measured attenuation correction,3d scatter correction,relative scatter scaling,decay correction,frame-length correction,randoms smoothing}\n');
fprintf(fid,'method of attenuation correction:=measured\n');
fprintf(fid,'%%CT coverage:=on\n');
fprintf(fid,'method of scatter correction:=Model-based\n');
fprintf(fid,'%%method of random correction:=dlyd\n');
fprintf(fid,'%%TOF mashing factor:=1\n');
fprintf(fid,'number of energy windows:=%d\n',SinoInfo.NumberOfEnergyWindows);
fprintf(fid,'%%energy window lower level (keV) [1]:=%f\n',SinoInfo.EnergyWindowLowerLevelKev1);
fprintf(fid,'%%energy window upper level (keV) [1]:=%f\n',SinoInfo.EnergyWindowUpperLevelKev1);
fprintf(fid,'%%coincidence window width (ns):=%0.4f\n',SinoInfo.CoincidenceWindowWidthNs);
fprintf(fid,'\n');
fprintf(fid,'!IMAGE DATA DESCRIPTION:=\n');
fprintf(fid,'!total number of data sets:=1\n');
fprintf(fid,'!image duration (sec):=%f\n',SinoInfo.ImageDurationSec);
fprintf(fid,'!image relative start time (sec):=%f\n',SinoInfo.ImageRelativeStartTimeSec);
fprintf(fid,'total prompts:=%d\n',SinoInfo.TotalPrompts);
fprintf(fid,'%%total randoms:=%d\n',SinoInfo.TotalRandoms);
fprintf(fid,'%%total net trues:=%d\n',SinoInfo.TotalNetTrues);
fprintf(fid,'%%GIM loss fraction:=%f\n',SinoInfo.GimLossFraction);
fprintf(fid,'%%PDR loss fraction:=%f\n',SinoInfo.PdrLossFraction);
fprintf(fid,'%%total uncorrected singles rate:=%d\n',SinoInfo.TotalUncorrectedSinglesRate);
fprintf(fid,'%%image slope:=1\n');
fprintf(fid,'%%image intercept:=0.0\n');
fprintf(fid,'maximum pixel count:=%f\n',ImgInfo.MinMaxImgCount(2));
fprintf(fid,'minimum pixel count:=%f\n',ImgInfo.MinMaxImgCount(1));
fprintf(fid,'\n');
fprintf(fid,'%%SUPPLEMENTARY ATTRIBUTES:=\n');
fprintf(fid,'%%axial compression:=%d\n',NormInfo.AxialCompression);
fprintf(fid,'%%maximum ring difference:=%d\n',NormInfo.MaximumRingDifference);

if isfield(ImgInfo,'scaleFactor')
    fprintf(fid,'%%BQML scale factor:=%f\n',ImgInfo.scaleFactor);
end

fclose(fid);