clear;clc
addpath('Measures');    %����ָ�꺯��λ��
%% Load data set
load Arts.mat;

Y = train_target';      %��ѵ�����ת��

%% Set MLRKELM para
parameter.C = 1;        %�������
parameter.Kpara = 1;    %RBF�˲���

[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = MLRKELM(train_data,Y,test_data,test_target,parameter);