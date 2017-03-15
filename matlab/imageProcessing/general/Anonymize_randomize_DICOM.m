
function Anonymize_randomize_DICOM(Path, patient_code)

% Path: the directory containing multiple DICOM folders
% patient_code: the code used in combination of a random number to replace
% dicom file name, patient name and SeriesDescription

% output an excel sheet with Anonymized-randomized codes and original file names


if(strcmp(computer(), 'GLNXA64')) % If linux, call wine64.
    bar = '/';
    os = 'linux';
else
    bar = '\';
    os = 'windows';
end

d = dir(Path);
noDicomFolders = sum([d(:).isdir])-2;

randNumbers = randperm(noDicomFolders,noDicomFolders);
NameCodes = cell(noDicomFolders,2);
k = 0;
for i = 3: length(d)
    if d(i).isdir
        k = k +1;
        subPathName = [Path bar d(i).name];
        RandomizedPatientCode = [num2str(randNumbers(k)) '-' patient_code];
        subPathAnonymizedName = [Path bar 'Anonymized' bar RandomizedPatientCode];
        mkdir(subPathAnonymizedName)
        di = dir(subPathName);
        for j = 3:length(di)
            hdr=dicominfo([subPathName bar di(j).name]);
            hdr.SeriesDescription = RandomizedPatientCode;
            hdr.PatientName.FamilyName = RandomizedPatientCode;
            img = dicomread(hdr);
            warning ('off')
            
            dicomwrite(img,[subPathAnonymizedName bar RandomizedPatientCode '-00' num2str(j-2) '.IMA'], hdr,'CreateMode','Copy' );
        end
        NameCodes{k,1} = d(i).name;
        NameCodes{k,2} = RandomizedPatientCode;
    end
    
end

xlswrite([Path bar 'Anonymized' bar 'Anonymized_Names.xlsx'],{'Code','Name'},'E5:F5');

xlsRange1 = ['E6:E' num2str(noDicomFolders+5)];
xlsRange2 = ['F6:F' num2str(noDicomFolders+5)];
xlswrite([Path bar 'Anonymized' bar 'Anonymized_Names.xlsx'],NameCodes(:,2),xlsRange1);
xlswrite([Path bar 'Anonymized' bar 'Anonymized_Names.xlsx'],NameCodes(:,1),xlsRange2);

if strcmpi(os,'windows')
    winopen([Path bar 'Anonymized' bar 'Anonymized_Names.xlsx'])
end


