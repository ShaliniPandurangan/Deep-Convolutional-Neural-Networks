warning off;
clear all;
close all;
clc;

% IMAGE ACQUISITION

Uiget=uigetfile('*.jpg;*.png','Pick an Image');
Image=imread(Uiget); %Read the Image into MATLAB
figure,imshow(Image);
title('Input Image');

% Intensity Image 
[r c d]=size(Image); %Calculate Row Column Dimention (Layers) of the Image

if d>2
    Image2=rgb2gray(Image); %If input image is RGB, then convert it into Gray format to calculate the intensity of the input image
else
    Image2=Image;
end

Ia1=Image2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% IMAGE DENOISING OR RESTORATION

% Adaptive Anisotropic  Filter
Iterations = 5; %Number of Iterations to be denoised
DiffusionPara = 30;
Image2 = double(Image2); %Double Recision ... to improve the MATLAB Memory
AnistoDiffuse = Image2;
Imo=str2double(Uiget(1:end-4));
% Center pixel distances.
X_Distance = 1; %Spacing between X and Y Directional coordinates
Y_Distance = 1; %Spacing between X and Y Directional coordinates
dd = sqrt(2); %Scaling Factor 

% 2D convolution masks - finite differences - Impulse Response coefficients
% to do convolutional filter
Mask1 = [0 1 0; 0 -1 0; 0 0 0];
Mask2 = [0 0 0; 0 -1 0; 0 1 0];
Mask3 = [0 0 0; 0 -1 1; 0 0 0];
Mask4 = [0 0 0; 1 -1 0; 0 0 0];
Mask5 = [0 0 1; 0 -1 0; 0 0 0];
Mask6 = [0 0 0; 0 -1 0; 0 0 1];
Mask7 = [0 0 0; 0 -1 0; 1 0 0];
Mask8 = [1 0 0; 0 -1 0; 0 0 0];

% Anisotropic diffusion.
for t = 1:Iterations

        % performs multidimensional filtering using convolution
        Filter1 = imfilter(AnistoDiffuse,Mask1,'conv');
        Filter2 = imfilter(AnistoDiffuse,Mask2,'conv');   
        Filter3 = imfilter(AnistoDiffuse,Mask4,'conv');
        Filter4 = imfilter(AnistoDiffuse,Mask3,'conv');   
        Filter5 = imfilter(AnistoDiffuse,Mask5,'conv');
        Filter6 = imfilter(AnistoDiffuse,Mask6,'conv');   
        Filter7 = imfilter(AnistoDiffuse,Mask7,'conv');
        Filter8 = imfilter(AnistoDiffuse,Mask8,'conv'); 
        
        Diffusion1 = 1./(1 + (Filter1/DiffusionPara).^2);
        Diffusion2 = 1./(1 + (Filter2/DiffusionPara).^2);
        Diffusion3 = 1./(1 + (Filter3/DiffusionPara).^2);
        Diffusion4 = 1./(1 + (Filter4/DiffusionPara).^2);
        Diffusion5 = 1./(1 + (Filter5/DiffusionPara).^2);
        Diffusion6 = 1./(1 + (Filter6/DiffusionPara).^2);
        Diffusion7 = 1./(1 + (Filter7/DiffusionPara).^2);
        Diffusion8 = 1./(1 + (Filter8/DiffusionPara).^2);
        
        % Discrete Partial Differential Function to Reconstruct the Image
        AnistoDiffuse = AnistoDiffuse + 0.1429*((1/(Y_Distance^2))*Diffusion1.*Filter1 + (1/(Y_Distance^2))*Diffusion2.*Filter2 +(1/(X_Distance^2))*Diffusion3.*Filter3 + (1/(X_Distance^2))*Diffusion4.*Filter4 + ...
                  (1/(dd^2))*Diffusion5.*Filter5 + (1/(dd^2))*Diffusion6.*Filter6 +(1/(dd^2))*Diffusion7.*Filter7 + (1/(dd^2))*Diffusion8.*Filter8 );           
end

AnistoDiffuse=uint8(ceil(AnistoDiffuse));
Ie2=AnistoDiffuse;
AnistoDiffuse = Conve(AnistoDiffuse);
figure,imshow(AnistoDiffuse/2);
title('Filtered Image using Anisotropic Diffusion Filter');

Ke=imsubtract(uint8(Ia1),uint8(Ie2));
figure,imshow(Ke*15);
title('Noise Content');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
x=uint8(Image2);
p=uint8(AnistoDiffuse);
si=size(x);
m=si(1);
n=si(2);
x=double(x);
p=double(p);
mse=0;
for i=1:m
    for j=1:n
    mse=mse+(x(i,j)-p(i,j))^2;
    end
end
U=str2double(Uiget(1:end-4));
mse=mse/(m*n);
psn=25*log10((255^2)/mse);
disp('Peak Signal to Noise Ratio using Anisotropic Diffusion Filter ...');
disp(psn);
disp('Mean Square Error using Anisotropic Diffusion Filter ...');
disp(mse);

% IMAGE ENHANCEMENT

% Input Image enhancement using Mean adjustment (Mean adjustment is used to
% improve the contrast, brightness)

% Intensity Thresholding
MediumThresholding=0.5; %Medium Thresholding limit .5
LowerThresholding=0.008; %Lower Thresholding limit 0
UpperThresholding=0.992; %Upper thresholding limit 1
AMF=AnistoDiffuse;
[R C D]=size(AMF); %If D=3 Multispectral Image, If D=1 Panchromic image

if D==3 %Condition for Image
    ColorImageUpperThreshold=0.04; %Bandwidth of the image 0.05 (+)
    ColorImageLowerThreshold=-0.04;%Bandwidth of the image -0.05 (-)
%     National Television System Committee ... Which contains Luminance
    NTSC=rgb2ntsc(AMF); %Standard color format (National Television Standard Color)
    MeanAdjust=ColorImageUpperThreshold-mean(mean(NTSC(:,:,2)));%Green Layer
    NTSC(:,:,2)=NTSC(:,:,2)+MeanAdjust*(0.596-NTSC(:,:,2));
    MeanAdjust=ColorImageLowerThreshold-mean(mean(NTSC(:,:,3))); %Blue Layer
    NTSC(:,:,3)=NTSC(:,:,3)+MeanAdjust*(0.523-NTSC(:,:,3));
else
    NTSC=double(AMF)./255; %All the image class is uint8 (unsigned Integer 8) 2^8=256.. there is linear variation
%     from 0 to 255) ... InputImage should be converted from uint8 to double
% '.'Scalar Product
end
% Mean Adjustment on the PAN Input Image 
MeanAdjust=MediumThresholding-mean(mean(NTSC(:,:,1)));
NTSC(:,:,1)=NTSC(:,:,1)+MeanAdjust*(1-NTSC(:,:,1)); %Mean adjustment formula
if D==3
    NTSC=ntsc2rgb(NTSC);
end
AMF=NTSC.*255; %uint8

%--------------------calculation of Minima and Maxima of the Mean adjusted InputImage ----------------------
for k=1:D
    Sort=sort(reshape(AMF(:,:,k),R*C,1)); %Convert the MxN matrix into column Matrix (Mx1)
    Minima(k)=Sort(ceil(LowerThresholding*R*C)); %Calculate the Minima
    Maxima(k)=Sort(ceil(UpperThresholding*R*C));%Calculate the maxima
end
%----------------------------------------------------------------------
if D==3
    Minima=rgb2ntsc(Minima);
    Maxima=rgb2ntsc(Maxima);
end
%----------------------------------------------------------------------
AMF=(AMF-Minima(1))/(Maxima(1)-Minima(1));%Ganzolez book 'Fundamentals of Digital Image Processing'
Enhancement=uint8(AMF.*255);
figure,imshow(Enhancement);
title('Contrast Enhanced InputImage'); 
[ra ca da]=size(Enhancement);
if da>2
    Enhancement=rgb2gray(Enhancement);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% SEGMENTATION
% MODIFIED FCM CLUSTERING SEGMENTATION
[Row Col]=size(Enhancement);
R=Row;
C=4;                            %Initialize the Class. K indicates that Weight Index for clustering
Enhancement=double(Enhancement);            %Double Precision of the image
NwIm=Enhancement;         
Enhancement=Enhancement(:);                 %Convert M x N Matrix into M x 1 Matrix in order to simplify the mathematical process      
MiPix=min(Enhancement);               %Calculate the Minima of the pixel       
Enhancement=Enhancement-MiPix+1;            %Calculate the posteriori probability using Baye's Rule  
Len=length(Enhancement);              %Calculate the length of the matrix
Histx=max(Enhancement)+1;             %Estimate the Maxima of the image pixel...Calculate the Histogram of the maxima pixel by adding 1
Buff=zeros(1,Histx);            %Initiate Empty (Zero) Buffer 1
Buffc=zeros(1,Histx);           %Initiate Empty (Zero) Buffer 2

for i=1:Len
    if(Enhancement(i)>0) 
        Buff(Enhancement(i))=Buff(Enhancement(i))+1; %Estimate intensity pixels by comparing the Histogram of each pixels
    end
end
LocPix=find(Buff);                  %Find the Location of the intensity pixels
Buffl=length(LocPix);               %Length of the Intensity Pixels
Cent=(1:C)*Histx/(C+1);             %Calculate the centre point using Histogram

while(1)  
    ReNm=Cent;
    % Classification Process starts here
    for i=1:Buffl
        Ab=abs(LocPix(i)-Cent);       %Calculate the absolute centroid position between central point and the Location of the intensity pixels
        Fib=find(Ab==min(Ab));        %Find the Minima of the above difference
        Buffc(LocPix(i))=Fib(1);      %Classification pixels
    end
    
    % Mean Estimation
    for i=1:C
        a=find(Buffc==i);             %Find the location of classified pixels
        Cent(i)=sum(a.*Buff(a))/sum(Buff(a)); %Calculate the mean of the classified pixels
    end
    
    if(Cent==ReNm)
        break;
    end
end
% calculate OutImg
Len=size(NwIm);         %Estimate the size of the image
OutImg=zeros(Len);      %Initiate the buffer
for i=1:Len(1)
    for j=1:Len(2)
        Ab=abs(NwIm(i,j)-Cent); %Calculate the absolute value of the image
        Comp=find(Ab==min(Ab)); %Find the location 
        OutImg(i,j)=Comp(1);    %Output image
    end
end
figure,imshow(OutImg,[])
title('Clustered Image');
vv=(OutImg);
Ox=zeros(r,c);
Mx=max(max(vv));
for i=1:r
    for j=1:c
        if vv(i,j)==Mx;
            Ox(i,j)=vv(i,j);
        end
    end
end
load Y;
Enhancement=Ox;
Enhancement=double(Enhancement);
[Rx,Cx]=size(Enhancement);
Ie=cat(3,Enhancement,Enhancement);
THL=8;
THH=254;
R1=repmat(THL,Rx,Cx);
R2=repmat(THH,Rx,Cx);
Re=cat(3,R1,R2);
R12=repmat(0.01,Rx,Cx);
Ree=cat(3,R12,R12);
Diff=Ie-Re;
Diff=Diff.*Diff+Ree;
Reci=1./Diff;
RediD=Reci(:,:,1)+Reci(:,:,2);
Dx1=Diff(:,:,1).*RediD;
Rex=1./Dx1;
Nrex=Diff(:,:,2).*RediD;
Rep=1./Nrex;
Pw=cat(3,Rex,Rep);
FCM=zeros(Rx,Cx);
if U>=8
    [ri ci d]=size(Image2);
    H=zeros(ri,ci);
    figure,imshow(H);
    title('Modified Fuzzy C Means Segmentation');
    return;
end
O=Y{:,U};
for i=1:Rx
    for j=1:Cx
        if Enhancement(i,j)>0
            FCM(i,j)=254;            
        else
            FCM(i,j)=0;
        end
    end
end
Fcc=FCM;
se = strel('line',3,2);
FCM = imerode(FCM,se);
figure,imshow(FCM,[]);
title('Modified Fuzzy C Means Segmentation');
figure,imshow(O);
title('Segmented BMD');
Oe=O.*255;
On(:,:,1)=Oe;
On(:,:,2)=Oe;
On(:,:,3)=Oe;
Imnew=imadd(double(Image),On);
figure,imshow(uint8(Imnew))
title('Superimposition of input and output images');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% FEATURE EXTRACTION
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
disp('Feature Extraction for Segmented Image');
echo on;
%           Autocorrelation
%           amplitude
%           Correlation
%           Correlation
%           Cluster Prominence   
%           Cluster Shade
%           Dissimilarity
%           Energy
%           Entropy
%           Homogeneity
%           Maximum probability
%           Sum of Squares
%           Sum Average
%           Sum Variance
%           Sum Entropy
%           Difference variance
%           Difference entropy
%           Information measures of correlation (1)
%           Information measures of correlation (2)
%           Maximal correlation coefficient
%           Inverse difference normalized (INN)
%           Inverse difference moment normalized (IDN)
echo off;
Ln=0;
for i=1:r
    for j=1:c
        if O(i,j)==Mx;
            Ln=Ln+1;
        end
    end
end
FEA = graycomatrix(abs(O),'Offset',[2 0;0 2]);
statsL = TextureFeatureExtraction(FEA,0);
disp('GLCM Feature Extraction');
disp(statsL);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
Features=[Ln statsL.autoc(1) statsL.contr(1) statsL.corrm(1) statsL.corrp(1) ...
    statsL.cprom(1) statsL.cshad(1) statsL.dissi(1) statsL.homom(1) statsL.entro(1) statsL.energ(1) ...
    statsL.maxpr(1) statsL.sosvh(1) statsL.savgh(1) statsL.svarh(1) statsL.senth(1) statsL.dvarh(1) ...
    statsL.denth(1) statsL.inf1h(1) statsL.inf2h(1) statsL.indnc(1) statsL.idmnc(1) statsL.autoc(2) ...
    statsL.contr(2) statsL.corrm(2) statsL.corrp(2) statsL.cprom(2) statsL.cshad(2) statsL.dissi(2) ...
    statsL.homom(2) statsL.entro(2) statsL.energ(2) statsL.maxpr(2) statsL.sosvh(2) statsL.savgh(2) ...
    statsL.svarh(2) statsL.senth(2) statsL.dvarh(2) statsL.denth(2) statsL.inf1h(2) statsL.inf2h(2) ...
    statsL.indnc(2) statsL.idmnc(2)];
Parameters=regionprops(O,'Area')
Parameters=regionprops(O,'Eccentricity')
Parameters=regionprops(O,'Perimeter')
Parameters=regionprops(O,'Solidity')
Mean=mean(mean(O))
Std=std(std(O))
Entropy=entropy(O)