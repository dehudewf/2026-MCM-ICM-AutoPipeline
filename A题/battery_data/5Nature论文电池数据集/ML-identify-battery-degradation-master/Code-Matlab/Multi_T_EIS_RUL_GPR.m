%Mulit-Temperature EIS-RUL GPR model tested on 35°C (Fig. 4(b))

%load relevant files
clear all;
filenames = {'EIS_data.txt', 'RUL.txt','EIS_data_RUL.txt','EIS_data_35C02.txt','rul35C02.txt'};
for kk = 1:numel(filenames)
    load(filenames{kk});
end

%Training set of the GRP model 
mean = mean(EIS_data,1); std = std(EIS_data,1);      %EIS_data is the raw experimental EIS data of the training cells cycled at 25, 35 and 45°C. EIS spectra are collected at state V.
X_train_rul = (EIS_data_RUL-mean)./std;              %X_train_rul is the input of the model after normalization
Y_train_rul = RUL;                                   %remaining useful life of each cell in the training, which is defined as the cycle number when the capacity drops below its initial 80%

%Mulit-Temperature EIS-RUL GPR model
meanfunc = @meanZero; hyp.mean = [];                 %mean function is zero
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);    %the Gaussian likelihood
covfunc = @covLINiso; hyp.cov = [0];                 %Linear covariance function
hyp_EIS_RUL = minimize(hyp, @gp, -10000, @infGaussLik, meanfunc, covfunc, likfunc, X_train_rul, Y_train_rul); %set hyperparameters by optimizing the (log) marginal likelihood

%Testing set of the GPR model
X_test_rul_35C02 = (EIS_data_35C02-mean)./std;           %EIS_data_35C02 is the EIS data of 35C02 cell.

%RUL prediction of the testing cell
[Y_test_rul_35C02,Y_test_rul_35C02_var] = gp(hyp_EIS_RUL,@infGaussLik,meanfunc,covfunc,likfunc,X_train_rul, Y_train_rul, X_test_rul_35C02); %Y_test_rul_35C02 is the predicted RUL. Y_test_rul_35C02_var is the uncertainty.

%The plot of the predicted RUL vs. Actual RUL (Fig. 4(b))
figure;
f = [flipud(Y_test_rul_35C02(1:127))+sqrt(Y_test_rul_35C02_var(1:127)); flipdim(flipud(Y_test_rul_35C02(1:127))-sqrt(Y_test_rul_35C02_var(1:127)),1)];
h=fill([[0:2:252]'; flipdim([0:2:252]',1)], f, [193 221 198]/255);
set(h,'LineStyle','none');set(gcf,'color','w');
hold on; plot(rul35C02,Y_test_rul_35C02(1:127),'hexagram','color',[119 172 45]/255,'LineWidth',1,'MarkerSize',6,'markerfacecolor',[119 172 45]/255);
xlim([0 252]);
ylim([0 300]); 
xlabel('\fontsize{25}Actual RUL');
ylabel('\fontsize{25}Predicted RUL');
title ('\fontsize{25}35C02');
