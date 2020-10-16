clear;clc
addpath('Measures');    %评价指标函数位置
%% Load data set
load Arts.mat;

Y = train_target';      %将训练标记转置

%% Set MLRKELM para
parameter.C = 1;        %正则参数
parameter.Kpara = 1;    %RBF核参数

[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = MLRKELM(train_data,Y,test_data,test_target,parameter);