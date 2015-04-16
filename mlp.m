function outTest=mlp(inputs, targets, testIn, nhidden, eta, nIteration, beta, momentum)
    %preparation phase
    dimIn=size(inputs);
    dimOut=size(targets);
    nin=dimIn(1,2);
    nout=dimOut(1,2);
    ndata=dimIn(1,1);
    weights1=(rand(nin+1,nhidden)-0.5)*2/sqrt(nin);
    weights2=(rand(nhidden+1,nout)-0.5)*2/sqrt(nhidden);
    inputs=horzcat(inputs,-ones(ndata,1));
    updatew1=zeros(size(weights1));
    updatew2=zeros(size(weights2));
    %training phase
    for n=1:nIteration
        %forward phase
        hidden=inputs*weights1;
        hidden=1 ./ (1+exp(-beta*hidden));
        hidden=horzcat(hidden,-ones(ndata,1));
        outputs=hidden*weights2;
        outputs=1 ./ (1+exp(-beta*outputs));
        %backward phase
        error=norm(outputs-targets);
         if mod(n,100)==0
             n
             error
         end
        deltao=beta*(outputs-targets).*outputs.*(1-outputs);
        deltah=beta*hidden.*(1-hidden).*(deltao*weights2');
        updatew1=eta * inputs' * deltah(:,1:end -1) + momentum * updatew1;
        updatew2=eta * hidden' * deltao + momentum * updatew2;
        weights1 = weights1 - updatew1;
        weights2 = weights2 - updatew2;
    end
    
    %calculate the output for testIn
    dimTest = size(testIn);
    ntest = dimTest(1,1);
    %testIn = horzcat(testIn,-ones(ntest,1))
    tstIn = horzcat(testIn,-ones(ntest,1));
    hiddenTest = tstIn*weights1;
    hiddenTest=1 ./ (1+exp(-beta*hiddenTest));
    hiddenTest=horzcat(hiddenTest,-ones(ntest,1));
    outTest = hiddenTest*weights2;
    outTest=1 ./ (1+exp(-beta*outTest));
end