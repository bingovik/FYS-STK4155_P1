from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import seaborn as sns
import inspect
import pdb

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def standardize(X):
    means = np.mean(X, axis=0)
    stdevs = np.std(X, axis=0)
    #attempt to leave bias column unchanged if included
    ind_zero_stdev = stdevs == 0
    stdevs[ind_zero_stdev] = 1
    means[ind_zero_stdev] = 0
    X_standardized = (X - means)/stdevs
    return X_standardized, means, stdevs

def OLS_analytical(X, y):
    try:
        beta = np.linalg.inv(X.T@X)@X.T@y
    except np.linalg.LinAlgError as err:
        beta = np.linalg.pinv(X)@y
    return beta

def Ridge_analytical(X, y, _lambda):
    I = np.eye(X.shape[1])
    try:
        beta = np.linalg.inv(X.T@X + _lambda*I)@X.T@y
    except np.linalg.LinAlgError as err:
        beta = np.linalg.pinv(X.T@X + _lambda*I)@X.T@y
    return beta

def OLS_scikitlearn(X, y):
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X, y)
    return clf.coef_

def Ridge_scikitlearn(X, y, _lambda):
    #clf = linear_model.Ridge(alpha=_lambda, fit_intercept=False)
    #clf.fit(X, y)
    #return clf.coef_
    clf = linear_model.Ridge(alpha=_lambda, fit_intercept=True)
    clf.fit(X, y)
    return np.hstack((clf.intercept_,clf.coef_[1:]))

def Lasso(X, y, _lambda):
    #clf = linear_model.Lasso(alpha=_lambda, max_iter=10**7, fit_intercept=False)
    #clf.fit(X, y)
    #return clf.coef_
    clf = linear_model.Lasso(alpha=_lambda, max_iter=10**7, fit_intercept=True)
    clf.fit(X, y)
    return np.hstack((clf.intercept_,clf.coef_[1:]))
    
def get_stats(y, y_predict):
    MSE = mean_squared_error(y, y_predict)
    R2 = r2_score(y, y_predict)
    return MSE, R2

def get_var_beta_Ridge(X,sigma,_lambda):
    I = np.eye(X.shape[1])
    try:
        var_beta = sigma**2*np.diag(np.linalg.inv(X.T@X+_lambda*I))*X.T*X*np.linalg.inv(X.T@X+_lambda*I).T
    except np.linalg.LinAlgError as err:
        var_beta = sigma**2*np.diag(np.linalg.pinv(X.T@X+_lambda*I))*X.T*X*np.linalg.pinv(X.T@X+_lambda*I).T
    return var_beta

def get_var_beta_OLS(X,sigma):
    try:
        var_beta = sigma**2*np.diag(np.linalg.inv(X.T@X))
    except np.linalg.LinAlgError as err:
        var_beta = sigma**2*np.diag(np.linalg.pinv(X.T@X))
    return var_beta

def surfPlot(x, y, z, xlabel = 'x', ylabel = 'y', zlabel = 'z', savefig = False, figname = ''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylabel(ylabel, fontsize = 9)
    ax.set_zlabel(zlabel, fontsize = 9)
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight') 
    plt.show()

def heatmap(data, title, xlabel, ylabel, xticks, yticks, annotation, savefig = False, figname = ''):
    ax = sns.heatmap(np.around(data, decimals=3), annot = annotation, linewidth=0.5)
    sns.set(font_scale=0.56)
    ax.set_title(title, fontsize = 11)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylabel(ylabel, fontsize = 9)
    ax.set_xticklabels(xticks, rotation=90, fontsize = 9)
    ax.set_yticklabels(yticks, rotation=0, fontsize = 9)
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight') 
    plt.show()

def plot_several(x_data, y_data, colors, labels, xlabel, ylabel, title, savefig = False, figname = ''):
    fig, ax = plt.subplots()
    plt.xlabel(xlabel, fontsize = 9)
    plt.ylabel(ylabel, fontsize = 9)
    ax.set_title(title, fontsize = 11)    
    for i in range(x_data.shape[1]):
        plt.plot(x_data[:,i], y_data[:,i], label = labels[i])
    leg = ax.legend()
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight') 
    plt.show()

def bootstrap_bias_variance_MSE(reg_func, X, y, n_boostraps, X_test, y_test, _lambda=0):
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(X, y)
        # Fit model and then evaluate the new model on the same test data each time.
        if len(inspect.getargspec(reg_func).args)==3:
            beta = reg_func(x_, y_, _lambda)
        else:
            beta = reg_func(x_, y_)
        y_pred[:, i] = X_test@beta
    error = np.mean( np.mean((y_test[:,None] - y_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (y_test[:,None] - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    return error, bias, variance

def cv(reg_func, X, y, k, X_test, y_test, _lambda=0):
    X_test_orig = X_test
    k_size = int(np.floor(len(X)/k))
    MSE_val = np.zeros(k)
    R2_val = np.zeros(k)
    y_predict_cv_val = np.zeros((k_size,k))
    y_predict_cv_test = np.zeros((len(X_test),k))
    y_cv_val = np.zeros((k_size,k))

    for i in range(k):
        test_ind = np.zeros(len(X), dtype = bool)
        test_ind[i*k_size:(i+1)*k_size] = 1
        X_cv_train = X[~test_ind]
        X_cv_val = X[test_ind]
        y_cv_train = y[~test_ind]
        y_cv_val[:,i] = y[test_ind]
        X_cv_train, mu, stdev = standardize(X_cv_train)
        X_cv_val = (X_cv_val - mu)/stdev
        X_test = (X_test_orig - mu)/stdev
        if len(inspect.getargspec(reg_func).args)==3:
            beta = reg_func(X_cv_train, y_cv_train, _lambda)
        else:
            beta = reg_func(X_cv_train, y_cv_train)
        y_predict_cv_val[:,i] = X_cv_val@beta
        y_predict_cv_test[:,i] = X_test@beta
        R2_val = r2_score(y_cv_val[:,i], y_predict_cv_val[:,i])
        MSE_val = mean_squared_error(y_cv_val[:,i], y_predict_cv_val[:,i])
    
    MSE_val = np.mean(MSE_val)
    R2_val = np.mean(R2_val)   
    MSE_test = np.mean(np.mean((y_test[:,None] - y_predict_cv_test)**2, axis=0, keepdims=True))    
    variance_test = np.mean(np.var(y_predict_cv_test, axis=1))
    bias_test_plus_noise = np.mean((y_test[:,None] - np.mean(y_predict_cv_test, axis=1, keepdims=True))**2)

    return MSE_val, MSE_test, R2_val, bias_test_plus_noise, variance_test