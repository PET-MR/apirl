
function ImgSUV = SUV(objGpet,Img,sinogramInterFileFilename,normalizationInterFileFilename)

[Img,info] = objGpet.BQML(Img,sinogramInterFileFilename,normalizationInterFileFilename);


Dose_at_injection_time = info.S.TracerActivityAtTimeOfInjectionBq;

Time = info.S.TracerInjectionTimeHhMmSsGmt0000;
injection_time = 3600*(str2double(Time(1:2)))+ 60*(str2double(Time(4:5)))+ 1*(str2double(Time(7:end)));

Time = info.S.StudyTimeHhMmSsGmt0000;
acquisition_time = 3600*(str2double(Time(1:2)))+ 60*(str2double(Time(4:5)))+ 1*(str2double(Time(7:end)));

TracerHalfLife = info.S.IsotopeGammaHalflifeSec;

Dose_at_acquisition_time = Dose_at_injection_time *exp(-log(2)/TracerHalfLife*(acquisition_time - injection_time));

% get PatientWeight from \e7\data-Converted\JSRecon12Info.txt or the dicom sinogram file in \e7\data
FromJSRecon = 1;
e7FileAddress = sinogramInterFileFilename(1:strfind(sinogramInterFileFilename,'e7')+2);
if FromJSRecon
    fid = fopen([e7FileAddress 'data-Converted\JSRecon12Info.txt'],'r');
    content = fread(fid,inf,'*char');
    p = strfind(content','PatientWeight');
    PatientWeight = str2double(content(p+[19:22])');
else
    dataFolder = [e7FileAddress 'data\'];
    d = dir(dataFolder);
    [~,i] = max([d.bytes]); % the largest dicom file is the sinogram file
    sinogramFilename = [dataFolder d(i).name];
    Info = dicominfo(sinogramFilename);
    PatientWeight = Info.PatientWeight;
end
ImgSUV = Img *(PatientWeight*1000) / Dose_at_acquisition_time;




