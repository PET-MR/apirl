
apirlPath = 'C:\MatlabWorkSpace\apirl-rw\matlab\andrew_reader_lab_software_interface\';
addpath(genpath('mMR_MrImageRecon'))
%% read MR rawdata, calculate coil sensitity maps, do retrospective undersampling
opt.mri_DataPath = '\\bioeng202-pc\PET-M\FDG_Patient_02\rawMPRAGE\meas_MID00074_FID59555_Head_t1_mprage_sag_AC_Lyon.dat';
opt.mri_DicomPath = '\\bioeng202-pc\PET-M\FDG_Patient_02\MPRAGE_image\'; % Dicom MRI imagea are used to calculate coordinates of the MRI space, using MRI rawdata for this is on progress 
opt.PhaseUnderSamplingFactor = 4;
opt.SliceUnderSamplingFactor = 1;

MRI = MRIReconClass(opt);

%% SENSE-CG reconstruction of fully sampled data
opt.ReconUnderSampledkSpace = 0;
opt.SENSE_niter =10;
opt.display = 100;

sense_f0 = MRI.SENSE_PCG2(opt);
[sense_f, RefSense_f ] = MRI.mapMrNativeSpaceToReferenceSpace(abs(sense_f0));
%% SENSE-CG reconstruction of undersampled data
opt.ReconUnderSampledkSpace = 1;
opt.SENSE_niter =20;
sense_u = MRI.SENSE_PCG(opt);
sense_u = MRI.mapMrNativeSpaceToReferenceSpace(abs(sense_u));

%% TV regualrized SENSE MRI image reconstruction
opt.PenaltyParameter = 1;
opt.RegualrizationParameter = 1000;
opt.ADMM_niter = 30;
opt.SENSE_niter = 2;
sense_u_tv = MRI.SENSE_TV_ADMM(opt);
sense_u_tv = MRI.mapMrNativeSpaceToReferenceSpace(abs(sense_u_tv));


%% Modify undersampling factor
opt.PhaseUnderSamplingFactor = 2;
opt.SliceUnderSamplingFactor = 2;
MRI.Revise(opt);


