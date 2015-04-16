function closest=knn(trainedData,dataClass,inputs,k)
    if nargin<4
        k=1;
    end
    dimTrain=size(trainedData);
    dimIn = size(inputs);
    nTrain = dimTrain(1,1);
    nData = dimIn(1,1);
    closest = zeros(nData,1);
    for n = 1:nData
        %calculate the distance from nth input data point to each trained
        %point
       distances=zeros(1,nTrain);
       dif=zeros(size(trainedData));
       for m = 1:nTrain
          dif(m,:) = trainedData(m,:)-inputs(n,:);
          distances(m) = norm(dif(m,:)); 
       end
       %sort and find the top k shortest distances
       [SortValue,SortIndex]=sort(distances);
       tarIndex=SortIndex(1:k);
       count1=0;
       count0=0;
       for i= 1:k
          if dataClass(tarIndex(i))==1 
              count1=count1+1;
          else
              count0=count0+1;
          end
       end
       closest(n)= count1>count0;
       %=========if more than two classes, use code below=======
%        %find the unique classes in such k points
%        tarClass=unique(dataClass(tarIndex));
%        count = zeros(size(tarClass));
%        %count for the frequency of each unique class
%        for i = 1:k
%            count = count+(tarClass==dataClass(SortIndex(i)));
%        end
%        %choose the most frequent unique class as the result for this input
%         [M,I]=max(count);
%         closest(n) = tarClass(I);
    end
end