%Mulit-Temperature EIS-Capacity GPR model tested on 35°C (Fig. 3(a))

%load relevant files
clear all;
filenames = {'EIS_data.txt', 'Capacity_data.txt','EIS_data_35C02.txt','capacity35C02.txt'};
for kk = 1:numel(filenames)
    load(filenames{kk});
end

%Training set of the GRP model 
mean = mean(EIS_data,1); std = std(EIS_data,1);      %EIS_data is the raw experimental EIS data of the training cells cycled at 25, 35 and 45°C. EIS spectra are collected at state V.
X_train = zscore (EIS_data);                         %X_train is the input of the model after normalization
Y_train = Capacity_data;                             %Capacity_data is the corresponding capacity of the training cells, defined as Y_train, the output of the model.

%Mulit-Temperature EIS-Capacity GPR model
meanfunc = @meanZero; hyp.mean = [];                 %mean function is zero
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);    %the Gaussian likelihood
covfunc = @covSEiso; hyp.cov = [0; 0];               %Squared Exponential covariance function
hyp_EIS_Capacity = minimize(hyp, @gp, -10000, @infGaussLik, meanfunc, covfunc, likfunc, X_train, Y_train); %set hyperparameters by optimizing the (log) marginal likelihood

%Testing set of the GPR model
X_test_35C02 = (EIS_data_35C02-mean)./std;           %EIS_data_35C02 is the EIS data of 35C02 cell.

%Capacity Estimation of the testing cell 
[Y_test_cap_35C02,Y_test_cap_35C02_var] = gp(hyp_EIS_Capacity,@infGaussLik,meanfunc,covfunc,likfunc,X_train, Y_train, X_test_35C02);  %Y_test_cap_35C02 is the estimated capacity. Y_test_cap_35C02_var is the uncertainty.

%The Plot of the estimated capacity (Fig. 3(a))
%The capacity is normalised against the starting capacity in each case.
figure;
f = [Y_test_cap_35C02/Y_test_cap_35C02(1,1)+sqrt(Y_test_cap_35C02_var)/Y_test_cap_35C02(1,1); flipdim(Y_test_cap_35C02/Y_test_cap_35C02(1,1)-sqrt(Y_test_cap_35C02_var)/Y_test_cap_35C02(1,1),1)]; 
h=fill([[2:2:598]'; flipdim([2:2:598]',1)], f, [255 191 200]/255);
set(h,'LineStyle','none');set(gcf,'color','w');
hold on; 
plot([2:2:598],capacity35C02/capacity35C02(1,1),'x','color', [0 130 216]/255,'LineWidth',3);
plot([2:2:598],Y_test_cap_35C02/Y_test_cap_35C02(1,1),'+','color',[205 39 70]/255,'LineWidth',3);
xlim([0 400]);
ylim([0.7 1.045]); 
xlabel('\fontsize{25}Cycle Number');
ylabel('\fontsize{25}Identified Capacity');
title ('\fontsize{25}35C02');
lgd = legend({'','\fontsize{25}Measured','\fontsize{25}Estimated'},'Box','off');

