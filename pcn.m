function [Weights,Activation]=pcn(Inputs, Targets, eta, nIteration)
    dimIn=size(Inputs);
    dimOut=size(Targets);
    nIn=dimIn(1,2);
    nOut=dimOut(1,2);
    nData=dimIn(1,1);
    Weights=rand(nIn+1,nOut)*0.1-0.05;
    Inputs = horzcat(-ones(nData,1),Inputs);
    Activation = (Inputs*Weights)>0;
    for i=1:nIteration
        Weights = Weights- eta*(Inputs'*(Activation-Targets));
        Activation = (Inputs*Weights)>0;
    end
end
