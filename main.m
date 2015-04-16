clear;clc;
%===================================================
%=================knn & mlp test====================
%===================================================
%   inputs1 = [0 0; 0 1; 1 0; 1 1];
%   targets1 = [0; 1; 1; 1];
%   inputs2 = [0 0; 0 1; 1 0; 1 1];
%   targets2 = [0; 1; 1; 0];
%   tIn = [1 0; 1 1; 0 1; 0 0];
%   test = [0.3 0; 0.8 0; 0.9 0.9; 0.1 0.2];
%   [W,A]=pcn(inputs2, targets2, 0.25, 10);
%   output=knn(inputs1,targets1,test)
%   mlp(inputs2,targets2,tIn, 3,0.25,50000,1,0.9);
%====================================================
%================end of test=========================
%====================================================


%=======================================================================
%read data files and store the original data into A0{i} for class 0,
%A1{i} for class 1.
%=======================================================================
maxNumData=20;
for i=1:maxNumData
    formatSpec0 = './outDataClass/%d_0.csv' ;
    formatSpec1 = './outDataClass/%d_1.csv' ;
    str0 = sprintf (formatSpec0,i);
    str1 = sprintf (formatSpec1,i);
    try
        A0{i}=csvread(str0,1,1);
    catch err
        disp(strcat(str0,', does not exist'));
    end
    
    try
        A1{i}=csvread(str1,1,1);
    catch err
        disp(strcat(str1,', dose not exist'));
    end
    
end

%===================================================================
%extract staticstic feature from original data to inputs and targets
%===================================================================
nClass0=20;
nClass1=19;
%==============try feature1: median=================================
inputs=zeros(39,2);
targets=vertcat(zeros(nClass0,1),ones(nClass1,1));
for i=1:39
    if i<=nClass0
        inputs(i,:)=mean(A0{i});
    else
        inputs(i,:)=mean(A1{i-nClass0});
    end
end
%===============try feature2: normalized============================
inputs2=zeros(39,52);
for i=1:39
    if i<=nClass0
        tempmean=mean(A0{i});
        temp1=[A0{i}(:,1)-tempmean(1,1) A0{i}(:,2)-tempmean(1,2)];
        tempstd=std(A0{i});
        temp2=[temp1(:,1)/tempstd(1,1) temp1(:,2)/tempstd(1,2)];
        inputs2(i,:)=[temp2(:,1)' temp2(:,2)'];
    else
        tempmean=mean(A1{i-nClass0});
        temp1=[A1{i-nClass0}(:,1)-tempmean(1,1) A1{i-nClass0}(:,2)-tempmean(1,2)];
        tempstd=std(A1{i-nClass0});
        temp2=[temp1(:,1)/tempstd(1,1) temp1(:,2)/tempstd(1,2)];
        inputs2(i,:)=[temp2(:,1)' temp2(:,2)'];
    end
end
%===============try feature3: discrete cosine transform===================
inputs3=zeros(39,52);
for i=1:39
    if i<=nClass0
        tempdct=dct(A0{i});
        tempmean=mean(tempdct);
        temp1=[tempdct(:,1)-tempmean(1,1) tempdct(:,2)-tempmean(1,2)];
        tempstd=std(tempdct);
        temp2=[temp1(:,1)/tempstd(1,1) temp1(:,2)/tempstd(1,2)];
        inputs3(i,:)=[temp2(:,1)' temp2(:,2)'];
    else
        tempdct=dct(A1{i-nClass0});
        tempmean=mean(tempdct);
        temp1=[tempdct(:,1)-tempmean(1,1) tempdct(:,2)-tempmean(1,2)];
        tempstd=std(tempdct);
        temp2=[temp1(:,1)/tempstd(1,1) temp1(:,2)/tempstd(1,2)];
        inputs3(i,:)=[temp2(:,1)' temp2(:,2)'];
    end
end

%==========parameters to be compute=========
sum1=zeros(1,20);
TP1=zeros(1,20);
TN1=zeros(1,20);
FP1=zeros(1,20);
FN1=zeros(1,20);
accuracy1=zeros(1,20);
precision1=zeros(1,20);
recall1=zeros(1,20);
sensitivity1=zeros(1,20);
specificity1=zeros(1,20);
F1=zeros(1,20);
FPR1=zeros(1,20);
sum2=zeros(1,20);
TP2=zeros(1,20);
TN2=zeros(1,20);
FP2=zeros(1,20);
FN2=zeros(1,20);
accuracy2=zeros(1,20);
precision2=zeros(1,20);
recall2=zeros(1,20);
sensitivity2=zeros(1,20);
specificity2=zeros(1,20);
F2=zeros(1,20);
FPR2=zeros(1,20);
%=================end=======================
%====================================================================
%iterate 20 times to compute statistic parameters
%this is to select best model based on those statistic parameters
%====================================================================
x=1;
for p=0.1:0.1:2
    nData=length(inputs3(:,1));
    rand_indx=randperm(nData);
    kfold=39;
    W=horzcat((2-p)*ones(39,26),p*ones(39,26));%try different combination of Diastolic and Systolic
    %==================================================
    %leave one out cross validation
    %==================================================
    for i=1:kfold
        %data preparation
        test_indx{i}=rand_indx(1+floor(nData/kfold)*(i-1):floor(nData/kfold)*i);
        %use feature 1
        trainIn{i}=inputs;
        %use feature 2
        %trainIn{i}=inputs2;
        %use feature 3
        %trainIn{i}=inputs3;
        %use feature 3 and combine feature inputs
        %trainIn{i}=inputs3.*W;
        trainTar{i}=targets;
        correct_targets{i}=targets(test_indx{i},:);
        trainTar{i}(test_indx{i},:)=[];
        testIn{i}=trainIn{i}(test_indx{i},:);
        trainIn{i}(test_indx{i},:)=[];
        
        %knn method
        classify_knn{i}=knn(trainIn{i},trainTar{i},testIn{i},3);
        
        if classify_knn{i}(1,1)==1 & correct_targets{i}(1,1)==1
            TP1(x) = TP1(x)+1;
        elseif classify_knn{i}(1,1)==0 & correct_targets{i}(1,1)==0
            TN1(x) = TN1(x)+1;
        elseif classify_knn{i}(1,1)==1 & correct_targets{i}(1,1)==0
            FP1(x) = FP1(x)+1;
        elseif classify_knn{i}(1,1)==0 & correct_targets{i}(1,1)==1
            FN1(x)= FN1(x)+1;
        end
        
        %mlp method
        
        %for feature 1 only
        %norm_trainIn{i}=[trainIn{i}(:,1)/max(trainIn{i}(:,1)) trainIn{i}(:,2)/max(trainIn{i}(:,2))];
        %norm_testIn{i}=[testIn{i}(:,1)/max(testIn{i}(:,1)) testIn{i}(:,2)/max(testIn{i}(:,2))];
        classify_mlp{i}=mlp(trainIn{i},trainTar{i},testIn{i},4,0.25,1000,1,0.8);
        
        if classify_mlp{i}(1,1)<=0.5 & correct_targets{i}(1,1)==1
            TP2(x) = TP2(x)+1;
        elseif classify_mlp{i}(1,1)>0.5 & correct_targets{i}(1,1)==0
            TN2(x) = TN2(x)+1;
        elseif classify_mlp{i}(1,1)<=0.5 & correct_targets{i}(1,1)==0
            FP2(x) = FP2(x)+1;
        elseif classify_mlp{i}(1,1)>0.5 & correct_targets{i}(1,1)==1
            FN2(x) = FN2(x)+1;
        end

    end
    accuracy1(x)=(TP1(x)+TN1(x))/39
    precision1(x)=TP1(x)/(TP1(x)+FP1(x));
    recall1(x)=TP1(x)/(TP1(x)+FN1(x));
    sensitivity1(x)=TP1(x)/(TP1(x)+FN1(x));%TPR
    specificity1(x)=TN1(x)/(TN1(x)+FP1(x));
    F1(x)=2*precision1(x)*recall1(x)/(precision1(x)+recall1(x));
    FPR1(x)=FP1(x)/(FP1(x)+TN1(x));
    comfmat1{x}=[TP1(x) FP1(x); FN1(x) TN1(x)];
    
    accuracy2(x)=(TP2(x)+TN2(x))/39
    precision2(x)=TP2(x)/(TP2(x)+FP2(x));
    recall2(x)=TP2(x)/(TP2(x)+FN2(x));
    sensitivity2(x)=TP2(x)/(TP2(x)+FN2(x));%TPR
    specificity2(x)=TN2(x)/(TN2(x)+FP2(x));
    F2(x)=2*precision2(x)*recall2(x)/(precision2(x)+recall2(x));
    FPR2(x)=FP2(x)/(FP2(x)+TN2(x));
    comfmat2{x}=[TP2(x) FP2(x); FN2(x) TN2(x)];
    x=x+1;
end

% %====================================================================
% %according to the result above, the best model is when _____
% %now use this model to see the final performance
% %run the code below seperately to generate ROC, precision, etc.
% %====================================================================
% p=0.8;
% x=8;
% W=horzcat((2-p)*ones(39,26),p*ones(39,26));
% TP2(x)=0;
% FP2(x)=0;
% TN2(x)=0;
% FN2(x)=0;
% iteration=1;
% while FP2(x)<1 & iteration<100
%     TP2(x)=0;
%     FP2(x)=0;
%     TN2(x)=0;
%     FN2(x)=0;
%     for i=1:kfold
%     %data preparation
%     test_indx{i}=rand_indx(1+floor(nData/kfold)*(i-1):floor(nData/kfold)*i);
%     trainIn{i}=inputs3.*W;
%     trainTar{i}=targets;
%     correct_targets{i}=targets(test_indx{i},:);
%     trainTar{i}(test_indx{i},:)=[];
%     testIn{i}=trainIn{i}(test_indx{i},:);
%     trainIn{i}(test_indx{i},:)=[];
%     
%     %mlp method
%     classify_mlp{i}=mlp(trainIn{i},trainTar{i},testIn{i},4,0.25,1000,1,0.8);
%     
%     if classify_mlp{i}(1,1)<=0.5 & correct_targets{i}(1,1)==1
%         TP2(x) = TP2(x)+1;
%     elseif classify_mlp{i}(1,1)>0.5 & correct_targets{i}(1,1)==0
%         TN2(x) = TN2(x)+1;
%     elseif classify_mlp{i}(1,1)<=0.5 & correct_targets{i}(1,1)==0
%         FP2(x) = FP2(x)+1;
%     elseif classify_mlp{i}(1,1)>0.5 & correct_targets{i}(1,1)==1
%         FN2(x) = FN2(x)+1;
%     end
%     
%     end
%     iteration = iteration+1;
% end
% 
% accuracy2(x)=(TP2(x)+TN2(x))/39
% precision2(x)=TP2(x)/(TP2(x)+FP2(x));
% recall2(x)=TP2(x)/(TP2(x)+FN2(x));
% sensitivity2(x)=TP2(x)/(TP2(x)+FN2(x));%TPR
% specificity2(x)=TN2(x)/(TN2(x)+FP2(x));
% F2(x)=2*precision2(x)*recall2(x)/(precision2(x)+recall2(x));
% FPR2(x)=FP2(x)/(FP2(x)+TN2(x));
% comfmat2{x}=[TP2(x) FP2(x); FN2(x) TN2(x)];
% 
% %------------------plot mlp ROC----------------------
% tarROC=cell2mat(correct_targets);
% outMLP=cell2mat(classify_mlp);
% j=1;
% for threshold=0:0.001:1%min(outMLP):0.0001:max(outMLP)
%     TP(j)=sum(outMLP<=threshold & tarROC==1);
%     FP(j)=sum(outMLP<=threshold & tarROC==0);
%     TN(j)=sum(outMLP>threshold & tarROC==0);
%     FN(j)=sum(outMLP>threshold & tarROC==1);
%     j=j+1;
% end
% TPR=TP./(TP+FN);
% FPR=FP./(FP+TN);
% ACC=(TP+TN)/39;
% plot(FPR,TPR);
% 
% p=polyfit(FPR,TPR,2);
% y=polyval(p,FPR);
% y=min(y,1);
% plot(FPR,y);
% 
% f=fit(FPR',y','smoothingspline');
% plot(f);



