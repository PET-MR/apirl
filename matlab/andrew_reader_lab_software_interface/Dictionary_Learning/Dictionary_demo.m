% Image denoising using Dictionary learning
addpath('C:\MatlabWorkSpace\A-MAP-PET\Dictionary_learning\ompbox10\')

Img = phantom(200)+rand(200)*0.3;

% Initialize Dic object
p.DU_algorithm     = 'ak-svd';
p.SC_TargetError   = 1;
p.SC_algorithm     = 'omp';
p.ImageSize        = [size(Img,1),size(Img,2),size(Img,3)];
p.PatchSize        = 12;
p.DL_niter         = 20;
Dic = Dictionary(p)


% Extract image patches
P = Dic.Patch(Img);

% Learn dictionary D
D = Dic.DictionaryLearning(P);
Dic.plot_dictionary(D)

% % Denoising by Sparse Coding
p.SC_TargetError  =1.3;
p.SC_algorithm     = 'omp';
Dic.Revise(p);

[~,Img_hat] = Dic.derivative_DL(Img,D);

subplot(121),imshow(Img,[0,1])
subplot(122),imshow(Img_hat,[0,1])


