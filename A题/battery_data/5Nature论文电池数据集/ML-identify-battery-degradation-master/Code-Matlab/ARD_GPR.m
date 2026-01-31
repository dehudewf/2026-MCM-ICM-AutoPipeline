%ARD-GRP model for 35°C (Fig. 3(c))

%load relevant files
clear all;
filenames = {'EIS_data_35.txt','Capacity_data_35.txt'};
for kk = 1:numel(filenames)
    load(filenames{kk});
end

%Training set of the ARD-GRP model 
X_train_ARD = zscore (EIS_data_35);            %EIS_data_35 is the EIS data of the cell cycled at 35°C. X_train is the input of the model after normalization
Y_train_ARD = Capacity_data_35;                %Capacity_data is the corresponding capacity of the training cell at 35°C, defined as Y_train, the output of the model.

%ARD-GPR model for 35°C
meanfunc = @meanZero; hyp.mean = [];                            %mean function is zero
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);               %the Gaussian likelihood
covfunc = @covSEard; sf = 1.0; hyp.cov = log(ones(121,1));      %Squared Exponential with Automatic Relevance Determination covariance function
hyp_ARD = minimize(hyp, @gp, -10000, @infGaussLik, meanfunc, covfunc, likfunc, X_train_ARD, Y_train_ARD); %set hyperparameters by optimizing the (log) marginal likelihood

%The plot of ARD results (Fig. 3(c))
sigmaL =hyp_ARD.cov(1:end-1);
sigmaL(1:120) = 10.^sigmaL(1:120);
weights = exp(-sigmaL); % Predictor weights
weights = weights/sum(weights);

figure('units','normalized','outerposition',[0 0 0.7 1]);
semilogx(weights,'bo','LineWidth',5);
set(gcf,'color','w');
xlabel('\fontsize{30}Predictor index');
ylabel('\fontsize{30}Predictor weight');
annotation('textbox',[0.8 0.75 0.3 0.15],'string',{'91^{st}'},'FontSize',30,'LineStyle','none');

