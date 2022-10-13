import pandas as pd
import numpy as np
import scipy as sc
import traceback

from collections import OrderedDict
from itertools import compress

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def sfa_linear_model(model, dataset, DV, IVs, forced_in=None, get_fitted=True):
    '''
    Perform regression on individual independent variables, with optional forced in variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param get_fitted: boolean whether to get fitted values
    :return: tuple with 'results' and 'fitted_values' dataframes
    '''

    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['Variable'] = pd.Series([], dtype='str')
    if forced_in is not None:
        col_info['Forced In'] = pd.Series([], dtype='str')

    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')
    if has_intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    col_info['Var Coef'] = pd.Series([], dtype='float')

    if forced_in is not None:
        for forced_var in forced_in:
            col_info[forced_var + " Coef"] = pd.Series([], dtype='float')

    col_info["Rsq"] = pd.Series([], dtype='float')
    col_info["Variance Explained"] = pd.Series([], dtype='float')
    if forced_in is not None:
        col_info["Variance Explained Beyond Forced In"] = pd.Series([], dtype='float')

    col_info["RMSE"] = pd.Series([], dtype='float')
    col_info["MAE"] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # If there are forced in variables, we run a regression with just the forced in variables
    if forced_in:

        model_dataset = dataset[[DV] + forced_in].dropna()
        kept_index = model_dataset.index.values

        X = model_dataset[forced_in]
        Y = model_dataset[DV]

        results = model.fit(X, Y)

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = "None"
        results_dict['Forced In'] = "|".join(forced_in)
        results_dict['# Obs'] = model_dataset.shape[0]
        results_dict['# Miss'] = len(Y) - model_dataset.shape[0]
        if has_intercept:
            results_dict['Intercept'] = results.intercept_
        results_dict['Var Coef'] = 0

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.coef_[results.feature_names_in_.tolist().index(forced_var)]

        predicted = model.predict(X)
        forced_in_fitted = predicted.copy()

        results_dict["Rsq"] = r2_score(Y, predicted)
        results_dict["Variance Explained"] = explained_variance_score(Y, predicted)
        if forced_in is not None:
            results_dict["Variance Explained Beyond Forced In"] = np.nan
        results_dict["RMSE"] = mean_squared_error(Y, predicted, squared=False)
        results_dict["MAE"] = mean_absolute_error(Y, predicted)

        output = pd.concat([output, pd.DataFrame(results_dict, index=[0])]).reset_index(drop=True)

        # Fitted values
        if get_fitted:
            fitted_values["ForcedInVars"] = np.nan
            fitted_values.loc[kept_index, "ForcedInVars"] = predicted

    # Loop through variables
    for IV in IVs:
        print("Working on {}, which is #{} out of {}".format(IV, IVs.index(IV) + 1, len(IVs)))

        if forced_in is not None:
            if IV in forced_in:
                print("Skipping this variable, since it is being forced in already")
                continue

        if forced_in is not None:
            model_dataset = dataset[[DV, IV] + forced_in]
        else:
            model_dataset = dataset[[DV, IV]]

        model_dataset = model_dataset.dropna()
        kept_index = model_dataset.index.values

        if forced_in is not None:
            X = model_dataset[[IV] + forced_in]
        else:
            X = model_dataset[[IV]]
        Y = model_dataset[DV]

        results = model.fit(X, Y)

        # Generate outputs
        results_dict = OrderedDict()

        results_dict['Variable'] = IV
        if forced_in is not None:
            results_dict['Forced In'] = "|".join(forced_in)
        results_dict['# Obs'] = model_dataset.shape[0]
        results_dict['# Miss'] = len(Y) - model_dataset.shape[0]

        if has_intercept:
            results_dict['Intercept'] = results.intercept_
        results_dict['Var Coef'] = results.coef_[results.feature_names_in_.tolist().index(IV)]

        if forced_in is not None:
            for forced_var in forced_in:
                results_dict[forced_var + " Coef"] = results.coef_[results.feature_names_in_.tolist().index(forced_var)]

        predicted = model.predict(X)

        results_dict["Rsq"] = r2_score(Y, predicted)
        results_dict["Variance Explained"] = explained_variance_score(Y, predicted)
        if forced_in is not None:
            results_dict["Variance Explained Beyond Forced In"] = 1 - np.nansum((Y - predicted) ** 2) / np.nansum(
                (Y - forced_in_fitted) ** 2)

        results_dict["RMSE"] = mean_squared_error(Y, predicted, squared=False)
        results_dict["MAE"] = mean_absolute_error(Y, predicted)

        output = pd.concat([output, pd.DataFrame(results_dict, index=[0])]).reset_index(drop=True)

        # Fitted values
        if get_fitted:
            fitted_values[IV] = np.nan
            fitted_values.loc[kept_index, IV] = predicted

    if get_fitted is False:
        fitted_values = None

    return output, fitted_values


### Wrapper functions for sfa_linear_model
def sfa_ols(dataset, DV, IVs, intercept=True, forced_in=None, get_fitted=True):
    model = linear_model.LinearRegression(fit_intercept=intercept)
    return sfa_linear_model(model, dataset, DV, IVs, forced_in, get_fitted)

def sfa_poisson(dataset, DV, IVs, intercept=True, forced_in=None, get_fitted=True):
    model = linear_model.PoissonRegressor(fit_intercept=intercept, alpha=0)
    return sfa_linear_model(model, dataset, DV, IVs, forced_in, get_fitted)



def lasso_linear_regression(dataset, DV, IVs, forced_in=None, intercept=True, alpha_list=None):
    '''
    Runs LASSO regression on data for variable selection purposes and generates outputs for all LASSO
    Perform OLS regression on individual independent variables, with optional forced in variables
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables
    :param forced_in: optional list of forced in variables. All variables named here will be forced in
    :param intercept: Boolean indicating whether of not to include intercept
    :param alpha_list: List of alpha penalty values
    :return: pandas dataset containing summary statistics and coefficients
    '''

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if forced_in is not None:
        if isinstance(forced_in, str):
            forced_in = [forced_in]
        # Remove any IVs from the IVs list if they are already being forced in
        IVs = list(set(IVs) - set(forced_in))

    if alpha_list is None:
        alpha_list = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25]

    # Set up the result table in Pandas
    col_info = OrderedDict()
    col_info['Alpha'] = pd.Series([], dtype='float')
    col_info['Variables'] = pd.Series([], dtype='str')
    col_info['Converged?'] = pd.Series([], dtype='bool')
    col_info['Rsq'] = pd.Series([], dtype='float')

    if intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')

    if forced_in is not None:
        vars = forced_in + IVs
    else:
        vars = IVs

    for var in vars:
        col_info["{} Coef".format(var)] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    scaler = StandardScaler()

    if forced_in is not None:
        model_dataset = dataset[[DV] + IVs + forced_in].dropna()
        model_dataset[IVs + forced_in] = scaler.fit_transform(model_dataset[IVs + forced_in])
        X = model_dataset[forced_in + IVs].copy()
    else:
        model_dataset = dataset[[DV] + IVs].dropna()
        model_dataset[IVs] = scaler.fit_transform(model_dataset[IVs])
        X = model_dataset[IVs].copy()

    Y = model_dataset[DV]

    if forced_in is not None:
        X[forced_in] = X[forced_in] * 1000

    for alpha in alpha_list:

        lasso = linear_model.Lasso(alpha=alpha, fit_intercept=intercept)

        fitted_model = lasso.fit(X, Y)

        results = OrderedDict()

        results['Alpha'] = alpha
        results['Converged?'] = fitted_model.n_iter_ < fitted_model.max_iter  # Assuming max number of iterations met means no convergence

        results['Rsq'] = fitted_model.score(X, Y)

        if intercept:
            results['Intercept'] = fitted_model.intercept_

        var_list = []
        for var in vars:
            var_coef = fitted_model.coef_[vars.index(var)]
            if var in forced_in:
                var_coef = var_coef*1000
            if var_coef != 0:
                var_list.append(var)
            results["{} Coef".format(var)] = var_coef

        results['Variables'] = ";".join(var_list)

        output = pd.concat([output, pd.DataFrame(results, index=[0])]).reset_index(drop=True)

    return output

default_mfa_options = {} # Not currently used
def mfa_linear_model(model, dataset, DV, IVs, get_fitted=True, detailed=False, mfa_options=default_mfa_options):
    '''
    Perform linear model regression on a set of independent variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IVs: list of names of independent variables to be evaluated one by one
    :param get_fitted: boolean whether to get fitted values
    :param detailed: boolean whether to produce detailed test results and charts. Currently set up to only produce VIF
    :param mfa_options: dictionary with test options. NOT CURRENTLY USED
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''

    # Check if intercept
    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IVs is None:
        raise ValueError("You must include IVs")

    if isinstance(IVs, str):
        IVs = [IVs]

    model_dataset = dataset[[DV] + IVs].dropna()
    kept_index = model_dataset.index.values

    X = model_dataset[IVs]
    Y = model_dataset[DV]

    results = model.fit(X,Y)
    predicted = model.predict(X)
    
    # Generate outputs
    results_dict = OrderedDict()

    for i in range(len(IVs)):
        IV = IVs[i]
        results_dict['Var {}'.format(i + 1)] = IV
        results_dict['Var {} Coef'.format(i + 1)] = results.coef_[results.feature_names_in_.tolist().index(IV)]

    if has_intercept:
        results_dict['Intercept'] = results.intercept_

    results_dict['# Obs'] = model_dataset.shape[0]
    results_dict['# Miss'] = len(Y) - model_dataset.shape[0]

    results_dict["Rsq"] = r2_score(Y, predicted)
    results_dict["Variance Explained"] = explained_variance_score(Y, predicted)
    results_dict["RMSE"] = mean_squared_error(Y, predicted, squared=False)
    results_dict["MAE"] = mean_absolute_error(Y, predicted)

    # Statistical tests
    VIF = sklearn_vif(X)
    results_dict["Max_VIF"] = max(VIF['VIF'])

    if detailed:
        detailed_results = OrderedDict()
        detailed_results['VIF'] = VIF
        regression_object = model

    else:
        detailed_results = None
        regression_object = None

    # Fitted values
    if get_fitted:
        fitted_values = pd.DataFrame({"kept_index": kept_index, "fit": predicted})

    if get_fitted is False:
        fitted_values = None

    return {
        "summary": results_dict,
        "fitted": fitted_values,
        "detailed": detailed_results,
        "model": regression_object
    }
    
    
def mfa_linear_models(model, dataset, DV, IV_table, get_fitted=True, detailed=False, mfa_options=default_mfa_options):
    '''
    Perform OLS regression on individual independent variables, with optional forced in variables
    
    Perform linear model regression on a set of independent variables
    :param model: Instantiated model class from https://scikit-learn.org/stable/modules/linear_model.html
    :param dataset: pandas dataframe
    :param DV: name of target variable (dependent variable)
    :param IV_table: pandas table where each row contains the variables 
    :param get_fitted: boolean whether to get fitted values
    :param detailed: boolean whether to produce detailed test results and charts. Currently set up to only produce VIF
    :param mfa_options: dictionary with test options. NOT CURRENTLY USED
    in a model
    :return: dictionary with 'results' and 'fitted_values' (if requested)
    '''
    
    # Check if intercept
    has_intercept = model.fit_intercept

    # Check inputs and reformat if necessary
    if DV is None:
        raise ValueError("You must include DV")
    if IV_table is None:
        raise ValueError("You must include IV table")

    num_var = IV_table.shape[1]

    # Set up the result table in Pandas
    col_info = OrderedDict()

    # Variable names
    for i in range(num_var):
        col_info['Var {}'.format(i+1)] = pd.Series([], dtype='str')

    col_info['# Obs'] = pd.Series([], dtype='int')
    col_info['# Miss'] = pd.Series([], dtype='int')

    # Coefficients
    if has_intercept:
        col_info['Intercept'] = pd.Series([], dtype='float')
    for i in range(num_var):
        col_info['Var {} Coef'.format(i+1)] = pd.Series([], dtype='float')
    
    # Model fit
    col_info["Rsq"] = pd.Series([], dtype='float')
    col_info["Variance Explained"] = pd.Series([], dtype='float')
    col_info["RMSE"] = pd.Series([], dtype='float')
    col_info["MAE"] = pd.Series([], dtype='float')
    
    # Tests
    col_info["Max_VIF"] = pd.Series([], dtype='float')

    # Create the pandas
    output = pd.DataFrame(col_info)

    # Create Pandas for fitted values
    fitted_values = dataset[[DV]].copy()
    fitted_values.columns = ["Actual"]

    # Loop through model table
    for i in range(IV_table.shape[0]):
    
        print("Working on model {} out of {}".format(i+1, IV_table.shape[0]))
        IV_list = IV_table.iloc[i,:]

        # Remove None and blanks
        IV_list = [x for x in IV_list if not pd.isnull(x)]
        IV_list = [x for x in IV_list if x != ""]
        
        try:
            # Execute main function
            reg_results = mfa_linear_model(model, dataset, DV, IV_list, get_fitted, detailed=False, mfa_options=mfa_options)
            
            # Add results to table
            output = pd.concat([output, pd.DataFrame(reg_results['summary'], index=[i])])
            
            # Fitted values
            if get_fitted:
                predictions = reg_results['fitted']
                fitted_values["Model {}".format(i+1)] = np.nan
                fitted_values.loc[predictions['kept_index'], "Model {}".format(i+1)] = predictions['fit']
        except:
            print("Model {} was not able to execute.".format(i+1))
            traceback.print_exc()

    if get_fitted is False:
        fitted_values = None

    return output, fitted_values
    



# Utilities -- for more statistical tests, use the statsmodels regression notebooks
def sklearn_vif(X_data):

    # Exogenous factors
    exogs = X_data.columns.tolist()
    
    # Initialize
    vifs = []

    if len(exogs) == 1:
        vifs = [1]
    
    else: 
        # form input data for each exogenous variable
        for exog in exogs:
            not_exog = [i for i in exogs if i != exog]
            X, y = X_data[not_exog], X_data[exog]

            # extract r-squared from the fit
            r_squared = linear_model.LinearRegression(fit_intercept=True).fit(X, y).score(X, y)

            # calculate VIF
            vif = 1/(1 - r_squared)
            vifs.append(vif)
        
    # return VIF DataFrame
    df_vif = pd.DataFrame({'Var': exogs, 'VIF': vifs})

    return df_vif



def plot_LassoCV_path(lasso_model, figsize=(10,6)):
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    
    ax.semilogx(lasso_model.alphas_, np.sqrt(lasso_model.mse_path_), ":")
    ax.plot(
        lasso_model.alphas_ ,
        np.sqrt(lasso_model.mse_path_.mean(axis=-1)),
        "k",
        label="Average across the folds",
        linewidth=2,
    )
    ax.axvline(
        lasso_model.alpha_, linestyle="--", color="k", label="alpha: estimate for each CV fold"
    )
    
    ax.legend()
    ax.set_xlabel("alphas")
    ax.set_ylabel("RMSE")
    ax.set_title("Root Mean Square Error");
    ax.axis("tight");

    table = pd.DataFrame({"Alpha": lasso_model.alphas_, "Mean RMSE": np.sqrt(lasso_model.mse_path_.mean(axis=-1))})
    
    return ax, table


# TODO: Add indicator for whether within X standard deviation from best result    
def plot_ElasticNetCV_path(ENet_model, figsize=(10,6), cmap=None):

    if ENet_model.alphas is None:
        # Alphas were automatically generated
        L1 = ENet_model.alphas_.copy()
        for i in range(L1.shape[0]):
            L1[i] = L1[i] * ENet_model.l1_ratio[i]

        # Check
        if any(L1.std(axis=0) > 0.0000001):
            raise ValueError("System detected that alpha values are automatically generated, yet couldn't find a consistent set of L1 penalties they correspond to")

        x_values = np.around(L1[0],4)
        x_label = "L1 penalty values (alpha * l1_ratio)"
        
    else:
        # Alphas were set manually
        x_values = ENet_model.alphas
        x_values.reverse()
        x_label = "Alpha values"
    
    frame = pd.DataFrame(np.sqrt(ENet_model.mse_path_.mean(axis=-1)), columns=x_values, index= ENet_model.l1_ratio)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax = sns.heatmap(frame, linewidth=0.5, ax=ax, cmap = cmap)
    ax.set_xlabel(x_label)
    ax.set_ylabel('l1_ratio')
    plt.show()

    return ax, frame
