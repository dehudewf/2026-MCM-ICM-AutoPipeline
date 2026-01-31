import os
import pandas as pd
import numpy as np
import shap
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer, max_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from xgboost import XGBRegressor

def describe(data, gt):
    # data = data.drop(data[data.CPU == 1].index)
    corr_matrix = data.corr()

    # Step 3: Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()


    corr_matrix = gt.drop(['sum_odpm','TPU_ENERGY_AVG_UWS'],axis=1).corr()

    # Step 3: Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

    df=gt.drop(['sum_odpm','TPU_ENERGY_AVG_UWS'],axis=1)
    corr_matrix = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    pvalues_matrix = pd.DataFrame(np.zeros_like(corr_matrix, dtype=float), columns=df.columns, index=df.columns)

    # 2. Calculate correlation coefficients and p-values for each pair using pearsonr
    for i in range(len(df.columns)):
        for j in range(i, len(df.columns)):
            # Calculate correlation and p-value
            corr, p_value = pearsonr(df.iloc[:, i], df.iloc[:, j])
            
            # Store the correlation and p-value in respective matrices
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr  # Symmetric matrix
            
            pvalues_matrix.iloc[i, j] = p_value
            pvalues_matrix.iloc[j, i] = p_value  # Symmetric matrix

    # 3. Replace non-significant correlations with NaN
    alpha = 0.05  # Significance level
    corr_matrix_with_nan = corr_matrix.copy()

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if pvalues_matrix.iloc[i, j] > alpha:  # Non-significant correlation
                corr_matrix_with_nan.iloc[i, j] = np.nan  # Replace with NaN

    # 4. Plot the heatmap, handling NaN values and adding annotations
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_with_nan, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, 
                mask=np.isnan(corr_matrix_with_nan), annot_kws={"size": 10})

    # Annotate dashes for non-significant correlations
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if pvalues_matrix.iloc[i, j] > alpha:
                plt.text(j + 0.5, i + 0.5, "-", ha='center', va='center', color='black', fontsize=12)

    plt.title("Correlation Heatmap with Non-Significant Correlations Replaced by Dash")
    plt.show()


    target = 'Battery_discharge_uWs'  # Replace with your target variable name
    correlations = data.corr()[target].drop(target)  # Drop the target itself

    # Step 3: Plot the correlations
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    #plt.title(f'Correlation of Features with {target}')
    plt.xlabel('Correlation Coefficient')
    plt.xlim([-0.2, 0.8])
    plt.ylabel('Features')
    plt.show()


    # target = 'CPU'  # Replace with your target variable name
    # gt['CPU'] = gt['CPU_BIG_ENERGY_AVG_UWS'] + gt['CPU_MID_ENERGY_AVG_UWS'] + gt['CPU_LITTLE_ENERGY_AVG_UWS']
    # d = data.merge(gt['CPU'],  left_index=True, right_index=True)
    # correlations = d.corr()[target].drop(target)  # Drop the target itself
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    # #plt.title(f'Correlation of Features with {target}')
    # plt.xlabel('Correlation Coefficient')
    # plt.xlim([-0.2, 0.8])
    # plt.ylabel('Features')
    # plt.title('CPU')
    # plt.show()




    # target = 'disp'  # Replace with your target variable name
    # gt['disp'] = gt['Display_ENERGY_AVG_UWS']
    # d = data.merge(gt['disp'],  left_index=True, right_index=True)
    # correlations = d.corr()[target].drop(target)  # Drop the target itself
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    # #plt.title(f'Correlation of Features with {target}')
    # plt.xlabel('Correlation Coefficient')
    # plt.xlim([-0.2, 0.8])
    # plt.ylabel('Features')
    # plt.title('disp')
    # plt.show()

    # target = 'wlan'  # Replace with your target variable name
    # gt['wlan'] = gt['WLANBT_ENERGY_AVG_UWS']
    # d = data.merge(gt['wlan'],  left_index=True, right_index=True)
    # correlations = d.corr()[target].drop(target)  # Drop the target itself
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
    # #plt.title(f'Correlation of Features with {target}')
    # plt.xlabel('Correlation Coefficient')
    # plt.xlim([-0.2, 0.8])
    # plt.ylabel('Features')
    # plt.title('wlan')
    # plt.show()

    print(data.describe())


    target = 'Battery_discharge_uWs' 
    features = data.columns.drop(target)  # All columns except the target
    # Step 3: Create subplots grid
    n_features = len(features)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_features + n_cols - 1) // n_cols  # Compute the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Step 4: Plot lmplots for each feature
    for i, feature in enumerate(features):
        sns.regplot(x=feature, y=target, data=data, ax=axes[i], scatter_kws={'s': 20})
        axes[i].set_title(f'{feature} vs power')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("power")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # sns.pairplot(data, kind="kde")
    # plt.show()

def average_error(X_test, y_test, y_pred, graph=False):
    abs_percentage_error1 = np.abs((y_test - y_pred) / y_test * 100)
    rel_percentage_error1 = (y_test - y_pred) / y_test * 100

    # for i, error in enumerate(abs_percentage_error):
    #     print(f"Sample {i+1}: Absolute Percentage Error = {error:.2f}%")
    data = pd.DataFrame({'abs': abs_percentage_error1, 'rel':rel_percentage_error1, 'actual': y_test, 'pred': y_pred})
    data = data.join(X_test)

    print(data.sort_values('abs',ascending=True).tail(10))
    if(graph):
        plt.figure(figsize=(5, 6)) 
        
        sns.violinplot(data=data['abs'], inner='stick')
        plt.ylabel("Absolute Percentage Error (%)")
        plt.show()

        sns.histplot(data=data['abs'])
        plt.xlabel("Absolute percentage error")
        plt.show()

        sns.lmplot(data=data, x="actual", y="rel")
        plt.xlabel("Measured energy consumption")
        plt.ylabel("Relative Percentage Errors")
        plt.show()

        sns.lmplot(data=data, x="actual", y="pred")
        plt.xlabel("Measured energy consumption")
        plt.ylabel("Relative Percentage Errors")
        plt.show()

    return data.describe()




def explain(X, model):
    if type(model) is RandomForestRegressor:
        importance = model.feature_importances_
    else:
        importance = model.coef_
        
    for i,v in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (X.columns[i],v))

    importance_df = pd.DataFrame({'Feature': X.columns,'Importance': importance})

    # Sort the DataFrame by the absolute value of the importance
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False)

    print(importance_df.tail(50))
    sns.barplot(x='Abs_Importance', y='Feature', data=importance_df)
    plt.xlabel('Abs_Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances (Sorted by Absolute Value)')
    plt.show()

def run(X_train, X_test, y_train, y_test, figures=False, gc=False):
    if gc:
        param_grid = {
            'alpha': np.logspace(-6, 6, 13),
            'l1_ratio': np.linspace(0.1, 1.0, 10),
            'max_iter': [500,1000, 5000, 10000],  # Increase iterations,
            'fit_intercept': [True],
            'positive': [True] 
        }
        grid_search = GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
    
        # Get the best model from the grid search
        model = grid_search.best_estimator_
        #best_params = grid_search.best_params_
        
        # Make predictions and calculate the mean squared error
    else:
        model = ElasticNet(alpha= 100.0, fit_intercept= True, l1_ratio=0.2, max_iter=1000, positive= True)
        model = ElasticNet(alpha=100.0, fit_intercept= True, l1_ratio= 0.2, max_iter= 1000, positive= True)
        model = Ridge(alpha= 10000.0, fit_intercept= True, positive= True, solver= 'auto')
        model = LinearRegression(positive=True, fit_intercept=False)
        ElasticNet(alpha=0.000006, fit_intercept= True, l1_ratio= 1, max_iter= 1000, positive= False)


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Training set score", model.score(X_train, y_train))
    print("Test set score", model.score(X_test, y_test))

    print(f"Model: {model.__class__.__name__} ")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"Mean Squared Error: {mse} \t \t R-squared: {r2} \t \t R-squared scores for each fold: {scores}")
    
    percentage_error = np.abs((y_pred - y_test) / y_test) * 100
    med = np.percentile(percentage_error, 50)
    
    #explain(X, model)
    
    #average_error(X_test, y_test, y_pred, figures)
    

    return (model, mse, r2, med)

def contribution(X, model):

    weights = model.coef_
    intercept = model.intercept_
    print(X)
    print(weights)
    print(intercept)
    net_contributions = X * weights
    print(net_contributions)
    epsilon = 1e-10
    total_contributions = np.sum(net_contributions, axis=1) + epsilon - intercept
    #total_contributions = np.sum(net_contributions, axis=1) + epsilon 
    # total_contributions = model.predict(X)
    # print('-----------------------')
    # print(total_contributions2-total_contributions)
    # print(total_contributions)
    # print(total_contributions2)
    net_percentage_contributions = ((net_contributions.T)/ total_contributions).T
    print(net_percentage_contributions)
    # print(net_contributions)
    # print("Percentage Contributions:\n", net_percentage_contributions)
    net_percentage_contributions.to_csv('contribution_percentage.csv',decimal=',')
    net_contributions.to_csv('contribution_value.csv',decimal=',')
    return net_percentage_contributions


def odpm_to_total(df):

    print(pearsonr(df['sum_odpm'],df['Battery_discharge_uWs']))
    df['error'] = df['sum_odpm']/df['Battery_discharge_uWs']
    correction = df['error'].mean()
    print('erreur :')
    print(correction)
    plt.figure(figsize=(5,5))
    sns.regplot(data=df, x="Battery_discharge_uWs", y="sum_odpm", scatter_kws={'s':2})

    plt.xlabel('Battery discharge speed (μW)')
    plt.ylabel('Sum of ODPM rails (μW)')
    #plt.title('Correlation of the sum of ODPM rails to the battery discharge')
    plt.tight_layout()
    plt.ylim(0, 8500000)
    plt.xlim(0, 8500000)
    plt.tight_layout()
    plt.savefig("figures/odpm_to_battery.pdf", format="pdf")
    # plt.show()


    plt.figure(figsize=(7,6))
    sns.boxplot(data=data[rails], orient='h', whis=(1, 99), showfliers=False)
    new_labels = [label.split('_ENERGY')[0] for label in rails]
    plt.gca().set_yticklabels(new_labels)

    plt.xlabel('Repartition of average power draw across iterations (μW)')
    plt.ylabel('Available ODPM rails')
    plt.savefig("figures/odpm_per_rail.pdf", format="pdf")
    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

    corr = data[rails].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()




def contribution_ground_truth(local, rails):
    for r in rails:
        local[r+'_odpm'] = local[r]/local['sum_odpm']
        local[r+'_bat'] = local[r]/local['Battery_discharge_uWs']
    local = local.drop(rails + ['sum_odpm','Battery_discharge_uWs'], axis=1)
    return local

    

def run_for_size(D, size, ground_truth_contribution, figures=False, gc=False, to_csv=False):
    D_subset = D.sample(n=size)
    X_sub = D_subset.drop(columns='Battery_discharge_uWs')
    y_sub = D_subset['Battery_discharge_uWs']
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2)

    model, mse, r2, med = run(X_train, X_test, y_train, y_test, figures=figures, gc=gc)
    
#        print(X_test.columns)
    estimated_contribution = contribution(X_test, model)
    estimated_contribution['sum_est_contrib'] = estimated_contribution.sum(axis=1)
    merged = pd.merge(estimated_contribution, ground_truth_contribution, left_index=True, right_index=True)

    merged['Display'] = (merged['RedLvl'] + merged['BlueLvl'] + merged['GreenLvl'] + merged['Brightness'])
    merged['Display_GT'] = merged['Display_ENERGY_AVG_UWS'] / merged['sum_odpm']
    merged['Dispay_error'] = np.abs(merged['Display'] - merged['Display_GT'])
    
    
    merged['CPU1'] = (merged['CPU_little']) 
    merged['CPU1_GT'] =(merged['CPU_LITTLE_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
    merged['CPU1_error'] = np.abs(merged['CPU1'] - merged['CPU1_GT'])

    merged['CPU2'] = (merged['CPU_mid']) 
    merged['CPU2_GT'] =(merged['CPU_MID_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
    merged['CPU2_error'] = np.abs(merged['CPU2'] - merged['CPU2_GT'])


    merged['CPU3'] = (merged['CPU_big']) 
    merged['CPU3_GT'] =(merged['CPU_BIG_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
    merged['CPU3_error'] = np.abs(merged['CPU3'] - merged['CPU3_GT'])


    merged['CPUa'] = (merged['CPU_big']  + merged['CPU_mid'] + merged['CPU_little']) 
    merged['CPUa_GT'] =(merged['CPU_BIG_ENERGY_AVG_UWS']+ merged['CPU_MID_ENERGY_AVG_UWS']  + merged['CPU_LITTLE_ENERGY_AVG_UWS']  + merged['Memory_ENERGY_AVG_UWS.1']+ merged['Memory_ENERGY_AVG_UWS'] + merged['S6M_LLDO1_ENERGY_AVG_UWS']+ merged['S8M_LLDO2_ENERGY_AVG_UWS']+ merged['S9M_VDD_CPUCL0_M_ENERGY_AVG_UWS']  ) / merged['sum_odpm']
    merged['CPUa_error'] = np.abs(merged['CPUa'] - merged['CPUa_GT'])

    merged['Net'] = merged['WIFI_data'] 
    merged['Net_GT'] = merged['WLANBT_ENERGY_AVG_UWS']  / merged['sum_odpm']
    merged['net_error'] = np.abs(merged['Net'] - merged['Net_GT'])

    merged['GPU'] = (merged['GPU0'] +merged['GPU1'] + merged['GPU_mem']) 
    merged['GPU_GT'] = (merged['GPU_ENERGY_AVG_UWS'] + merged['GPU3D_ENERGY_AVG_UWS']) / merged['sum_odpm']
    merged['GPU_error'] = np.abs(merged['GPU'] - merged['GPU_GT'])


    merged['all'] = merged['Display'] + merged['CPUa'] + merged['Net'] + merged['GPU']

    if(figures):
        sns.regplot(data=merged, x= 'CPUa_GT', y='CPUa')
        plt.title('CPU')
        plt.show()
        sns.regplot(data=merged, x= 'Net_GT', y='Net')
        plt.title('Net')
        plt.show()
        sns.regplot(data=merged, x= 'Display_GT', y='Display')
        plt.title('Display')
        plt.show()
        sns.regplot(data=merged, x= 'GPU_GT', y='GPU')
        plt.title('GPU')
        plt.show()

    # merged['Memory'] = (merged['M_ID']+merged['M_PL']) 
    # merged['M_GT'] = (merged['Memory_ENERGY_AVG_UWS.1']+ merged['Memory_ENERGY_AVG_UWS']) / merged['sum_odpm']
    # merged['M_error'] = np.abs(merged['Memory'] - merged['M_GT'])
    # , ('Memory','M_GT')
    # ('CPU1','CPU1_GT'), ('CPU2','CPU2_GT'), ('CPU3','CPU3_GT'),   
    results = []
    for (a,b) in [('Display','Display_GT'),('CPUa','CPUa_GT'),('Net','Net_GT'), ('GPU','GPU_GT')]:
        
        # Compute the required metrics for the current (a, b) pair
        avg_b = np.mean(merged[b]) * 100 # Average of ground truth for this group
        avg_a = np.mean(merged[a]) * 100 # Average of ground truth for this group
        diff_perc_abs = np.mean(np.abs(merged[a] - merged[b])/merged[b]) * 100
        diff_perc = np.mean((merged[a] - merged[b])/merged[b]) * 100
        avg_diff_abs = np.mean(np.abs(merged[a] - merged[b])) * 100 # Average absolute difference
        avg_diff = np.mean(merged[a] - merged[b]) * 100  # Average absolute difference
        correlation, p_value = spearmanr(merged[a], merged[b])  # Spearman correlation and p-value
        correlation2, p_value2 = pearsonr(merged[a], merged[b])  # Spearman correlation and p-value
        gt_cv = (merged[b].std()/merged[b].mean() ) *100
        
        # Store the results in a dictionary
        results.append({
            'group': a,
            'avg_gt': avg_b,
            'avg_e': avg_a,
            'diff_perc': diff_perc,
            'diff_perc_abs': diff_perc_abs,
            'avg_diff': avg_diff,
            'avg_diff_abs': avg_diff_abs,
            'gt_cv': gt_cv,
            'spearman_corr': correlation,
            'p_value_s': p_value,
            'pearson_corr': correlation2,
            'p_value_p': p_value2
        })
    r =pd.DataFrame(results).set_index('group')
    print(r)
    df = pd.DataFrame(results).set_index('group')
    # Convert the results list to a dataframe
    print(df)

    if(to_csv):
        df.to_csv('per_component_accuracy.csv')

    return r['spearman_corr'], mse, r2, med

def run_and_estimate(D, ground_truth_contribution, sizes=[10000], gc=False, figures=False, to_csv=False):
    df = pd.DataFrame()
    for i in sizes:
        d, mse, r2, med = run_for_size(D, i, ground_truth_contribution, figures=figures, gc=gc, to_csv=to_csv)
        print(d)
        local = d.to_frame().T
        print(local)
        local['size'] = i
        local['mse'] = mse
        local['r2'] = r2
        local['med'] = med
        df = pd.concat([df, local])

    

    return df.set_index('size')

def compare_models(X_train, X_test, y_train, y_test, figure=False, filename="model_comparison.csv"):
    # Define the models to compare
    models = [
        ("Linear Regression", LinearRegression(fit_intercept= True)),
        ("Ridge", Ridge(alpha= 0.000006, fit_intercept= True, positive= False, solver= 'svd')),
        ("Lasso", Lasso(alpha= 0.00001, fit_intercept= True, max_iter= 1000, positive= False, selection= 'random')),
        ("Elastic Net", ElasticNet(alpha=0.000006, fit_intercept= True, l1_ratio= 1, max_iter= 1000, positive= False)),
        ("Decision Tree", tree.DecisionTreeRegressor(max_depth= 10, min_samples_leaf= 5, min_samples_split= 2)),
        ("Random Forest", RandomForestRegressor(bootstrap= True, max_depth= None, min_samples_leaf=1, min_samples_split= 2, n_estimators= 200)),
        ("Gradient Boosting", GradientBoostingRegressor(learning_rate= 0.1, max_depth= 3, n_estimators= 200, subsample= 0.8) ),
        # ("SVR", SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        ("KNeighborsRegressor", KNeighborsRegressor(metric= 'manhattan', n_neighbors= 3, weights= 'distance')),
        # ("xgb", XGBRegressor(n_estimators=100))
    ]
    
    # List to store the comparison data
    comparison_data = []
    all_percentage_errors = []
    model_names = []

    # Loop over models, fit them, and calculate the error
    for name, model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate MSE, R-squared, and percentage error
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate absolute error as a percentage of y_test
        percentage_error = np.abs((y_pred - y_test) / y_test) * 100
        
        # Calculate quartiles, min, max, and average of percentage error
        q1 = np.percentile(percentage_error, 25)
        q2 = np.percentile(percentage_error, 50)  # Median
        q3 = np.percentile(percentage_error, 75)
        avg_error = np.mean(percentage_error)
        min_error = np.min(percentage_error)
        max_error = np.max(percentage_error)
        
        # Store the model's performance in the comparison data
        comparison_data.append({
            'Model': name,
            'MSE': mse,
            'R-squared': r2,
            'Min Error (%)': min_error,
            'Q1 (%)': q1,
            'Median (%)': q2,
            'Q3 (%)': q3,
            'Max Error (%)': max_error,
            'Average Error (%)': avg_error
        })
        
        # Store percentage errors for the plot
        all_percentage_errors.extend(percentage_error)
        model_names.extend([name] * len(percentage_error))
    
    # Create a DataFrame to save comparison data to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(filename, index=False)
    
    # Create the violin plot if figure=True
    if figure:
        # Create the DataFrame for the plot
        error_df = pd.DataFrame({
            'Model': model_names,
            'Percentage Error': all_percentage_errors
        })
        
        # Create the violin plot for the error rates
        plt.figure(figsize=(10, 4))
        sns.boxplot(x='Percentage Error', y='Model', data=error_df, orient='h', showfliers=False)
        # plt.title("Comparison of Percentage Error Rates")
        plt.xlabel("Percentage Error (%)")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig("figures/model_comparison.pdf", format="pdf")

        plt.show()

def compare_models_v2(X_train, X_test, y_train, y_test, ground_truth_contribution, figure=False, filename="model_comparison.csv"):
    # Define the models to compare
    models = [
        # ("Linear Regression1", LinearRegression(fit_intercept= True, positive=True)),
        # ("Linear Regression1", LinearRegression(fit_intercept= True, positive=False)),
        # ("Linear Regression1", LinearRegression(fit_intercept= False, positive=True)),
        # ("Linear Regression1", LinearRegression(fit_intercept= False, positive=False)),
        # ("Ridge", Ridge(alpha= 0.0001, fit_intercept= True, positive= True, solver= 'auto')),
        # ("Lasso", Lasso(alpha= 0.0001, fit_intercept= True, max_iter= 1000, positive= True, selection= 'random')),
        # ("Elastic Net", ElasticNet(alpha=0.0001, fit_intercept= True, l1_ratio= 0, max_iter= 1000, positive= True)),
        ("Elastic Net", ElasticNet(alpha=0.000001, fit_intercept= True, l1_ratio= 1, max_iter= 1000, positive= True)),
    ]
    
    # List to store the comparison data
    comparison_data = []
    all_percentage_errors = []
    model_names = []

    # Loop over models, fit them, and calculate the error
    for name, model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate MSE, R-squared, and percentage error
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate absolute error as a percentage of y_test
        percentage_error = np.abs((y_pred - y_test) / y_test) * 100
        
        # Calculate quartiles, min, max, and average of percentage error
        q1 = np.percentile(percentage_error, 25)
        q2 = np.percentile(percentage_error, 50)  # Median
        q3 = np.percentile(percentage_error, 75)
        avg_error = np.mean(percentage_error)
        min_error = np.min(percentage_error)
        max_error = np.max(percentage_error)
        
        # Store the model's performance in the comparison data
        
        # Store percentage errors for the plot
        all_percentage_errors.extend(percentage_error)
        model_names.extend([name] * len(percentage_error))
    
        if type(model) is RandomForestRegressor:
            importance = model.feature_importances_
        else:
            importance = model.coef_
        print(importance)
#        print(X_test.columns)
        estimated_contribution = contribution(X_test, importance, model.intercept_)
        estimated_contribution['sum_est_contrib'] = estimated_contribution.sum(axis=1)
        merged = pd.merge(estimated_contribution, ground_truth_contribution, left_index=True, right_index=True)

        merged['Display'] = (merged['RedLvl'] + merged['BlueLvl'] + merged['GreenLvl'] + merged['Brightness'])
        merged['Display_GT'] = merged['Display_ENERGY_AVG_UWS'] / merged['sum_odpm']
        merged['Dispay_error'] = np.abs(merged['Display'] - merged['Display_GT'])
        
        
        merged['CPU1'] = (merged['CPU_little']) 
        merged['CPU1_GT'] =(merged['CPU_LITTLE_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
        merged['CPU1_error'] = np.abs(merged['CPU1'] - merged['CPU1_GT'])

        merged['CPU2'] = (merged['CPU_mid']) 
        merged['CPU2_GT'] =(merged['CPU_MID_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
        merged['CPU2_error'] = np.abs(merged['CPU2'] - merged['CPU2_GT'])


        merged['CPU3'] = (merged['CPU_big']) 
        merged['CPU3_GT'] =(merged['CPU_BIG_ENERGY_AVG_UWS'] ) / merged['sum_odpm']
        merged['CPU3_error'] = np.abs(merged['CPU3'] - merged['CPU3_GT'])


        merged['CPUa'] = (merged['CPU_big']  + merged['CPU_mid'] + merged['CPU_little']) 
        merged['CPUa_GT'] =(merged['CPU_BIG_ENERGY_AVG_UWS']+ merged['CPU_MID_ENERGY_AVG_UWS']  + merged['CPU_LITTLE_ENERGY_AVG_UWS']  + merged['Memory_ENERGY_AVG_UWS.1']+ merged['Memory_ENERGY_AVG_UWS'] + merged['S6M_LLDO1_ENERGY_AVG_UWS']+ merged['S8M_LLDO2_ENERGY_AVG_UWS']+ merged['S9M_VDD_CPUCL0_M_ENERGY_AVG_UWS']  ) / merged['sum_odpm']
        merged['CPUa_error'] = np.abs(merged['CPUa'] - merged['CPUa_GT'])

        merged['Net'] = merged['WIFI_data'] 
        merged['Net_GT'] = merged['WLANBT_ENERGY_AVG_UWS']  / merged['sum_odpm']
        merged['net_error'] = np.abs(merged['Net'] - merged['Net_GT'])

        merged['GPU'] = (merged['GPU0'] +merged['GPU1'] + merged['GPU_mem']) 
        merged['GPU_GT'] = (merged['GPU_ENERGY_AVG_UWS'] + merged['GPU3D_ENERGY_AVG_UWS']) / merged['sum_odpm']
        merged['GPU_error'] = np.abs(merged['GPU'] - merged['GPU_GT'])


        merged['all'] = merged['Display'] + merged['CPUa'] + merged['Net'] + merged['GPU']
        sns.regplot(data=merged, x= 'CPUa_GT', y='CPUa')
        plt.title('CPU')
        plt.show()
        sns.regplot(data=merged, x= 'Net_GT', y='Net')
        plt.title('Net')
        plt.show()
        sns.regplot(data=merged, x= 'Display_GT', y='Display')
        plt.title('CPU')
        plt.show()
        sns.regplot(data=merged, x= 'GPU_GT', y='GPU')
        plt.title('CPU')
        plt.show()

        # merged['Memory'] = (merged['M_ID']+merged['M_PL']) 
        # merged['M_GT'] = (merged['Memory_ENERGY_AVG_UWS.1']+ merged['Memory_ENERGY_AVG_UWS']) / merged['sum_odpm']
        # merged['M_error'] = np.abs(merged['Memory'] - merged['M_GT'])
        # , ('Memory','M_GT')
# ('CPU1','CPU1_GT'), ('CPU2','CPU2_GT'), ('CPU3','CPU3_GT'), 
        results = []
        for (a,b) in [('Display','Display_GT'),('CPUa','CPUa_GT'),('Net','Net_GT'), ('GPU','GPU_GT'), ('all','sum_est_contrib')]:
            
            # Compute the required metrics for the current (a, b) pair
            avg_b = np.mean(merged[b]) * 100 # Average of ground truth for this group
            avg_a = np.mean(merged[a]) * 100 # Average of ground truth for this group
            avg_diff_abs = np.mean(np.abs(merged[a] - merged[b])) * 100 # Average absolute difference
            avg_diff = np.mean(merged[a] - merged[b]) * 100  # Average absolute difference
            correlation, p_value = spearmanr(merged[a], merged[b])  # Spearman correlation and p-value
            correlation2, p_value2 = pearsonr(merged[a], merged[b])  # Spearman correlation and p-value
            gt_cv = (merged[b].std()/merged[b].mean() ) *100
            
            # Store the results in a dictionary
            results.append({
                'group': a,
                'avg_gt': avg_b,
                'avg_e': avg_a,
                'avg_diff': avg_diff,
                'avg_diff_abs': avg_diff_abs,
                'gt_cv': gt_cv,
                'spearman_corr': correlation,
                'p_value_s': p_value,
                'pearson_corr': correlation2,
                'p_value_p': p_value2
            })
        r =pd.DataFrame(results).set_index('group')
        print(r)
        sum_diff = r['avg_diff_abs'].sum()
        comparison_data.append({
            'Model': name,
            'MSE': mse,
            'R-squared': r2,
            'Min Error (%)': min_error,
            'Q1 (%)': q1,
            'Median (%)': q2,
            'Q3 (%)': q3,
            'Max Error (%)': max_error,
            'Average Error (%)': avg_error,
            'sum_diff': sum_diff
        })
        

        # Convert the results list to a dataframe

    # Create a DataFrame to save comparison data to CSV
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)


    


def grid_search_all_models(X, y):
    # Define models and parameter grids
    models = {
        "LinearRegression": (LinearRegression(), {
            'fit_intercept': [True, False],
            'positive': [True,False]
        }),
        
        "Ridge": (Ridge(), {
            'alpha': np.logspace(-6, 6, 13),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
            'fit_intercept': [True, False],
            'positive': [True, False] 
        }),

        "Lasso": (Lasso(), {
            'alpha': np.logspace(-6, 6, 13),
            'max_iter': [1000, 5000, 10000],  # Increase iterations
            'selection': ['cyclic', 'random'],
            'fit_intercept': [True, False],
            'positive': [True, False] 
        }),

        "ElasticNet": (ElasticNet(), {
            'alpha': np.logspace(-6, 6, 13),
            'l1_ratio': np.linspace(0.1, 1.0, 10),
            'max_iter': [1000, 5000, 10000],  # Increase iterations,
            'fit_intercept': [True, False],
            'positive': [True, False] 
        }),

        "DecisionTreeRegressor": (tree.DecisionTreeRegressor(random_state=42), {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }),

        "RandomForestRegressor": (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'bootstrap': [True, False]
        }),

        "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }),

        # "SVR": (SVR(), {
        #     'C': [0.1, 1, 10],
        #     'epsilon': [0.01, 0.1, 0.2],
        #     'kernel': ['linear', 'rbf']
        # }),

        "KNeighborsRegressor": (KNeighborsRegressor(), {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }),

        # "XGBRegressor": (XGBRegressor(random_state=42), {
        #     'n_estimators': [50, 100, 200],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'max_depth': [3, 5, 7],
        #     'subsample': [0.8, 1.0],
        #     'colsample_bytree': [0.8, 1.0]
        # })
    }

    # Perform GridSearchCV for each model
    results = []
    for model_name, (model, param_grid) in models.items():
        print(f"\nPerforming GridSearchCV for {model_name}...")
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        
        # Fit the model
        grid_search.fit(X, y)

        # Print the best parameters and best score
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score (negative MSE) for {model_name}: {grid_search.best_score_}")
        results.append({
            "model": model_name,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        })
        
        # Optionally, evaluate the model on a test set if you have one
        # For demonstration, we'll just print the best score in the training set (cross-validation score)

    print("\nGrid search completed for all models.")
    results_df = pd.DataFrame(results)
    results_df.to_csv("grid_search_results.csv", index=False)


rails = ['L21S_VDD2L_MEM_ENERGY_AVG_UWS',
        'UFS(Disk)_ENERGY_AVG_UWS',
        'S12S_VDD_AUR_ENERGY_AVG_UWS',
        'Camera_ENERGY_AVG_UWS',
        'GPU3D_ENERGY_AVG_UWS',
        'Sensor_ENERGY_AVG_UWS',
        'Memory_ENERGY_AVG_UWS',
        'Memory_ENERGY_AVG_UWS.1',
        'Display_ENERGY_AVG_UWS',
        'GPS_ENERGY_AVG_UWS',
        'GPU_ENERGY_AVG_UWS',
        'WLANBT_ENERGY_AVG_UWS',
        'L22M_DISP_ENERGY_AVG_UWS',
        'S6M_LLDO1_ENERGY_AVG_UWS',
        'S8M_LLDO2_ENERGY_AVG_UWS',
        'S9M_VDD_CPUCL0_M_ENERGY_AVG_UWS',
        'CPU_BIG_ENERGY_AVG_UWS',
        'CPU_LITTLE_ENERGY_AVG_UWS',
        'CPU_MID_ENERGY_AVG_UWS',
        'INFRASTRUCTURE_ENERGY_AVG_UWS',
        'CELLULAR_ENERGY_AVG_UWS',
        'CELLULAR_ENERGY_AVG_UWS.1',
        'INFRASTRUCTURE_ENERGY_AVG_UWS.1',
        'TPU_ENERGY_AVG_UWS'
        ]

rails_discard = ['L21S_VDD2L_MEM_ENERGY_UW',
        'UFS(Disk)_ENERGY_UW',
        'S12S_VDD_AUR_ENERGY_UW',
        'Camera_ENERGY_UW',
        'GPU3D_ENERGY_UW',
        'Sensor_ENERGY_UW',
        'Memory_ENERGY_UW',
        'Memory_ENERGY_UW.1',
        'Display_ENERGY_UW',
        'GPS_ENERGY_UW',
        'GPU_ENERGY_UW',
        'WLANBT_ENERGY_UW',
        'L22M_DISP_ENERGY_UW',
        'S6M_LLDO1_ENERGY_UW',
        'S8M_LLDO2_ENERGY_UW',
        'S9M_VDD_CPUCL0_M_ENERGY_UW',
        'CPU_BIG_ENERGY_UW',
        'CPU_LITTLE_ENERGY_UW',
        'CPU_MID_ENERGY_UW',
        'INFRASTRUCTURE_ENERGY_UW',
        'CELLULAR_ENERGY_UW',
        'CELLULAR_ENERGY_UW.1',
        'INFRASTRUCTURE_ENERGY_UW.1',
        'TPU_ENERGY_UW'
        ]



data = pd.read_csv('./res_test/aggregated.csv', index_col=0)
data = data.rename(columns={"RougeMesuré": "RedLvl", "VertMesuré": "GreenLvl", "BleuMesuré":"BlueLvl"})
data = data.rename(columns={"CPU_LITTLE_FREQ_KHz": "CPU_little", "CPU_MID_FREQ_KHz": "CPU_mid", "CPU_BIG_FREQ_KHz":"CPU_big"})
data = data.rename(columns={"GPU0_FREQ": "GPU0", "GPU_1FREQ": "GPU1", "GPU_MEM_AVG": "GPU_mem"})
data = data.rename(columns={"TOTAL_DATA_WIFI_BYTES": "WIFI_data"})

#data = data[~data['CPU_mid'].isin(['err'])].apply(pd.to_numeric)
#data = data[~data['Brightness'].isin([181])]
#data = data.drop(data[data.CPU_mid == 'err'].index).apply(pd.to_numeric)
data['CPU_mid'] = data['CPU_mid'].replace('err', np.nan).apply(pd.to_numeric)
data['CPU_mid'].fillna(data['CPU_mid'].median(), inplace=True)

data[['C_ID','M_ID','WIFI_data']] = data[['C_ID','M_ID','WIFI_data']].replace(0, np.nan)
data['C_ID'].fillna(data['C_ID'].median(), inplace=True)
data['M_ID'].fillna(data['M_ID'].median(), inplace=True)
data['WIFI_data'].fillna(data['WIFI_data'].median(), inplace=True)

data = data.drop(rails_discard, axis=1)
# data = data.drop(data[data.Brightness == 181].index)

print(data)

# corr_matrix = data.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Heatmap')
# plt.show()

data['sum_odpm'] = data[rails].sum(axis=1)
data['Battery_discharge_uWs'] = data['BATTERY_DISCHARGE_RATE_UAS'] * 4.4
data=data.drop('BATTERY_DISCHARGE_RATE_UAS', axis=1)


# sns.regplot(data=data, x='WIFI_data', y='WLANBT_ENERGY_AVG_UWS')
# plt.show()

# sns.regplot(data=data, x='WLANBT_ENERGY_AVG_UWS', y='Battery_discharge_uWs')
# plt.show()

# sns.regplot(data=data, x='WIFI_data', y='CPU_big')
# plt.show()

print(spearmanr(data['WIFI_data'], data['WLANBT_ENERGY_AVG_UWS']))
print(pearsonr(data['WIFI_data'], data['CPU_LITTLE_ENERGY_AVG_UWS']))


# odpm_to_total(data.copy())
gt_contribution = data[rails + ['sum_odpm']].copy()
gt_contribution.to_csv('ground_truth.csv', decimal=',')

data_no_odpm = data.drop(rails, axis=1).drop('sum_odpm', axis=1).drop('BATTERY_DISCHARGE_TOTAL_UA',axis=1).drop(['AVG_SOC_TEMP','DIFF_SOC_TEMP','BATTERY__PERCENT','C_ID','C_PL','M_ID','M_PL'], axis=1)

# print('rgb+brightness to screen')
# print(pearsonr(data['RedLvl'], data['Display_ENERGY_AVG_UWS']))
# print(pearsonr(data['GreenLvl'], data['Display_ENERGY_AVG_UWS']))
# print(pearsonr(data['BlueLvl'], data['Display_ENERGY_AVG_UWS']))
# print(pearsonr(data['Brightness'], data['Display_ENERGY_AVG_UWS']))
# print('GPU')
# print(pearsonr(data['GPU0'],data['GPU3D_ENERGY_AVG_UWS'] + data['GPU_ENERGY_AVG_UWS']))
# print(pearsonr(data['GPU1'],data['GPU3D_ENERGY_AVG_UWS'] + data['GPU_ENERGY_AVG_UWS']))
# print(pearsonr(data['GPU_mem'],data['GPU3D_ENERGY_AVG_UWS'] + data['GPU_ENERGY_AVG_UWS']))
# print('CPUlittle')
# print(pearsonr(data['CPU_little'],data['CPU_LITTLE_ENERGY_AVG_UWS']))
# print('CPUmid')
# print(pearsonr(data['CPU_mid'],data['CPU_MID_ENERGY_AVG_UWS']))
# print('CPUbig')
# print(pearsonr(data['CPU_big'],data['CPU_BIG_ENERGY_AVG_UWS']))
# print('Wifi')
# print(pearsonr(data['WIFI_data'],data['WLANBT_ENERGY_AVG_UWS']))

print('-=========')
print(pearsonr(data['CPU_big'],data['WIFI_data']))


# print(gt_contribution.columns)
# print(data_no_odpm.columns)

print('data validation')
nan_rows = data_no_odpm[data_no_odpm.isna().any(axis=1)]
print(nan_rows)

# print(data_no_odpm.describe().T)
# print(data[rails].describe().T)


print(data.columns)
# describe(data_no_odpm, gt_contribution)

X = data_no_odpm.drop(['Battery_discharge_uWs'], axis=1)
y = data_no_odpm['Battery_discharge_uWs']


size_comparison = run_and_estimate(data_no_odpm, gt_contribution, [1000], gc=True, to_csv=True)

# size_comparison = run_and_estimate(data_no_odpm, gt_contribution, [50,100,200,300,400,500,600,700,800,900,1000], gc=True, figures=False)
# print(size_comparison)
# size_comparison.to_csv('per_components_1000.csv')
# size_comparison.to_csv('size_comparison.csv')
# sns.lineplot(data=size_comparison, x='size', y='r2', linewidth=2.5)
# plt.show()
#compare_given_size(X,y, 200,gt_contribution)

exit()

# Initialize a list to store the r2 values of each run
num_runs = 10


# Initialize a list to store the DataFrames
df_list = []

# Run the function `num_runs` times
for _ in range(num_runs):
    df =  run_and_estimate(data_no_odpm, gt_contribution, [50,100,200, 300, 400, 500, 600, 700, 800, 900, 1000], gc=True, figures=False)
# print(size_comparison
    df_list.append(df)  # Append the entire DataFrame to the list

# Get the columns to average by selecting the columns of the first DataFrame
# Assuming all DataFrames have the same columns, we get the column names from the first one
columns = df_list[0].columns

# Initialize an empty list to hold the averaged columns
averaged_columns = []

# Loop through each column and compute the average across all DataFrames
for col in columns:
    # Extract the column for all DataFrames
    column_data = [df[[col]] for df in df_list]
    
    # Concatenate the columns along axis=1 (stacking the DataFrames)
    column_concat = pd.concat(column_data, axis=1)
    
    # Compute the mean along the "third axis" (i.e., axis=1)
    column_avg = column_concat.mean(axis=1)
    
    # Store the averaged column
    averaged_columns.append(column_avg)

# Create a final DataFrame with all averaged columns
final_df = pd.concat(averaged_columns, axis=1)

# Optionally, set 'size' as the index if desired
final_df.index = df_list[0].index  # Assuming all DataFrames have the same 'size' index
final_df.columns = columns  # Set the column names as those of the original DataFrames

# Show the final DataFrame with the averaged values
print(final_df)



sns.lineplot(data=final_df, x='size', y='r2', linewidth=2.5)
plt.show()

#print(y.describe())

# grid_search_all_models(X,y)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# compare_models_v2(X_train, X_test, y_train, y_test, gt_contribution, True)
# compare_models(X_train, X_test, y_train, y_test, True)
# print("head", X_test.head())

