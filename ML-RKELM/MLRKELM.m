function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = MLRKELM(X,Y,Xt,Yt,parameter)
C = parameter.C;
Kpara = parameter.Kpara;

[OutputWeight,Omega_test,Y] = kelmtrain (X, Y, Xt, C, Kpara);
TY = kelmpredict (OutputWeight,Omega_test);

Outputs = TY';
Pre_Labels = sign(Outputs);
test_target = Yt;
HammingLoss=Hamming_loss(Pre_Labels,test_target);

RankingLoss=Ranking_loss(Outputs,test_target);
OneError=One_error(Outputs,test_target);
Coverage=coverage(Outputs,test_target);
Average_Precision=Average_precision(Outputs,test_target);