from project1_functions import *
import pdb
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from decimal import Decimal

# setting models and hyper parameters
poly_degree_max = 20
poly_degree_max_Lasso = 5
lambda_test_number = 15
lambda_tests = np.logspace(-7, 1, num=lambda_test_number)

savefigs = True

'''
# Load the terrain
zz = imread('SRTM_data_Norway_2.tif')
#zz = zz[zz.shape[0]-round(zz.shape[0]/64):-1,zz.shape[1]-round(zz.shape[1]/64):-1]
zz = zz[zz.shape[0]-round(zz.shape[0]/96):-1,zz.shape[1]-round(zz.shape[1]/96):-1]

x_select = range(0,zz.shape[1],64)
y_select = range(0,zz.shape[0],64)
zz = zz[y_select,:]
zz = zz[:,x_select]

x = np.arange(zz.shape[1])
y = np.arange(zz.shape[0])
yy, xx = np.meshgrid(x,y)
sigma = 1 #unknown.. could estimate with: diff*diff/(n-p-1).

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(zz, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
# set noise and number of data points for Franke function evaluation
seed(1)
x = np.sort(np.random.rand(40))
seed(1)
y = np.sort(np.random.rand(40))
xx, yy = np.meshgrid(x,y)
zz = FrankeFunction(xx,yy)
sigma = 0.1 #random noise std dev
seed(1)
noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
zz = zz + noise

fignamePostFix = '_Franke_40_01'

surfPlot(xx, yy, zz, savefig = savefigs, figname = 'surfDataRaw' + fignamePostFix)

# declaring variables
MSE = np.zeros(poly_degree_max)
MSE_scikitlearn = np.zeros(poly_degree_max)
R2 = np.zeros(poly_degree_max)
var_beta_list = []
bias_test = np.zeros(poly_degree_max)
MSE_test = np.zeros(poly_degree_max)
MSE_train = np.zeros(poly_degree_max)
MSE_cv_val_OLS = np.zeros(poly_degree_max)
MSE_cv_test_OLS = np.zeros(poly_degree_max)
R2_cv_val_OLS = np.zeros(poly_degree_max)
bias_cv_test_OLS = np.zeros(poly_degree_max)
bias_plus_noise_cv_test_OLS = np.zeros(poly_degree_max)
variance_cv_test_OLS = np.zeros(poly_degree_max)
MSE_bootstrap_OLS = np.zeros(poly_degree_max)
bias_bootstrap_OLS = np.zeros(poly_degree_max)
variance_bootstrap_OLS = np.zeros(poly_degree_max)
MSE_cv_val_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
R2_cv_val_Ridge = np.zeros((poly_degree_max, lambda_test_number))
bias_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
bias_plus_noise_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
variance_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_cv_val_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
MSE_cv_test_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
R2_cv_val_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
bias_cv_test_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
bias_plus_noise_cv_test_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
variance_cv_test_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
MSE_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_test_Lasso = np.zeros((poly_degree_max_Lasso, lambda_test_number))
MSE_bootstrap_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
bias_bootstrap_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
variance_bootstrap_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
list_of_features = []
var_beta_list = []
var_beta_list_Lasso = []
beta_Ridge = []
beta_Lasso = []
z_predict_Ridge = []

poly_degrees = np.arange(poly_degree_max)+1
lambda_tests_str = ['%.1E' % Decimal(str(lam)) for lam in lambda_tests]
z = np.ravel(zz)
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

# running regressions using different polynomial fits up to poly_degree_max
for poly_degree in poly_degrees:
    print('Polynomial degree: %g' % poly_degree)
    # creating polynomials of degree poly_degree
    poly = PolynomialFeatures(poly_degree) #inlude bias = false
    X = poly.fit_transform(X_orig)
    features = poly.get_feature_names(['x','y'])
    list_of_features.append(features)
    # feature scaling
    X, mu, stdev = standardize(X)
    
    # OLS on whole sample
    beta = OLS_analytical(X, z)
    z_predict = X@beta
    MSE[poly_degree - 1], R2[poly_degree - 1] = get_stats(z, z_predict)
    var_beta = get_var_beta_OLS(X, sigma)
    var_beta_list.append(list(var_beta))
    beta = OLS_scikitlearn(X, z) #try scikitlearn for comparison
    z_predict = X@beta
    MSE_scikitlearn[poly_degree - 1], _ = get_stats(z, z_predict)

    #surfPlot(xx, yy, z_predict.reshape((len(y),len(x))))

    # Splitting data into train and test sets
    seed(0)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)

    # feature scaling again
    X_train, mu, stdev = standardize(X_train)
    X_test = (X_test - mu)/stdev

    # OLS evaluated on test set
    beta = OLS_scikitlearn(X_train, z_train)
    z_predict_train = X_train@beta
    z_predict_test = X_test@beta
    MSE_train[poly_degree-1], R2_test = get_stats(z_train, z_predict_train)
    MSE_test[poly_degree-1], R2_train = get_stats(z_test, z_predict_test)

    # OLS bias and variance estimates using cv and bootstrap
    MSE_cv_val_OLS[poly_degree-1], MSE_cv_test_OLS[poly_degree-1], R2_cv_val_OLS[poly_degree-1], bias_plus_noise_cv_test_OLS[poly_degree-1], variance_cv_test_OLS[poly_degree-1] = cv(OLS_scikitlearn, X_train, z_train, 10, X_test, z_test)
    MSE_bootstrap_OLS[poly_degree-1], bias_bootstrap_OLS[poly_degree-1], variance_bootstrap_OLS[poly_degree-1] = bootstrap_bias_variance_MSE(OLS_scikitlearn, X_train, z_train, 100, X_test, z_test, _lambda=0)

    # Ridge and Lasso regression using cv and with different regularisation lambda
    for (i, lambda_test) in enumerate(lambda_tests):
        print('lambda %g' % lambda_test)
        
        # Ridge bias and variance estimates using cv and bootstrap
        MSE_cv_val_Ridge[poly_degree-1, i], MSE_cv_test_Ridge[poly_degree-1, i], R2_cv_val_Ridge[poly_degree-1, i], bias_plus_noise_cv_test_Ridge[poly_degree-1, i], variance_cv_test_Ridge[poly_degree-1, i] = cv(Ridge_scikitlearn, X_train, z_train, 10, X_test, z_test, lambda_test)
        MSE_bootstrap_test_Ridge[poly_degree-1,i], bias_bootstrap_test_Ridge[poly_degree-1,i], variance_bootstrap_test_Ridge[poly_degree-1,i] = bootstrap_bias_variance_MSE(Ridge_scikitlearn, X_train, z_train, 100, X_test, z_test, _lambda=lambda_test)

        #calculating Ridge MSE using whole training sample (not cross-validating)
        beta = Ridge_scikitlearn(X_train, z_train, lambda_test)

        #storing Ridge predictions
        z_predict_Ridge.append([])
        z_predict_Ridge[poly_degree-1].append([])
        z_predict_Ridge[poly_degree-1][i].append(X@beta)

        #storing Ridge betas
        beta_Ridge.append([])
        beta_Ridge[poly_degree-1].append([])
        beta_Ridge[poly_degree-1][i].append(beta)

        #predicting Ridge on test set
        z_predict_test_Ridge = X_test@beta
        MSE_test_Ridge[poly_degree-1, i] = mean_squared_error(z_test, z_predict_test_Ridge)

        if poly_degree <= poly_degree_max_Lasso:
            MSE_cv_val_Lasso[poly_degree-1, i], MSE_cv_test_Lasso[poly_degree-1, i], R2_cv_val_Lasso[poly_degree-1, i], bias_plus_noise_cv_test_Lasso[poly_degree-1, i], variance_cv_test_Lasso[poly_degree-1, i] = cv(Lasso, X_train, z_train, 10, X_test, z_test, lambda_test)
            #calculating Lasso MSE using whole training sample (not cross-validating)
            beta = Lasso(X_train, z_train, lambda_test)

            #storing Lasso betas
            beta_Lasso.append([])
            beta_Lasso[poly_degree-1].append([])
            beta_Lasso[poly_degree-1][i].append(beta)

            z_predict_test_Lasso = X_test@beta
            MSE_test_Lasso[poly_degree-1, i] = mean_squared_error(z_test, z_predict_test_Lasso)
        
min_ind_cv_val_Ridge = np.unravel_index(np.argmin(MSE_cv_val_Ridge, axis=None), MSE_cv_val_Ridge.shape)
min_ind_cv_test_Ridge = np.unravel_index(np.argmin(MSE_cv_test_Ridge, axis=None), MSE_cv_test_Ridge.shape)
min_ind_cv_val_OLS = np.argmin(MSE_cv_val_OLS)
min_ind_cv_test_OLS = np.argmin(MSE_cv_test_OLS)

print('Ridge best parameters: Degree=%g, Lambda=%g' % (poly_degrees[min_ind_cv_val_Ridge[0]],lambda_tests[min_ind_cv_val_Ridge[1]]))
z_predict_Ridge_min = np.asarray(z_predict_Ridge[min_ind_cv_val_Ridge[0]][min_ind_cv_val_Ridge[1]])[0]
surfPlot(xx, yy, z_predict_Ridge_min.reshape((len(y),len(x))), savefig = savefigs, figname = 'Ridge_best_val_Surf' + fignamePostFix)

# displaying OLS cross validation results
heatmap(np.hstack((MSE_cv_val_OLS[:,None],MSE_cv_test_OLS[:,None])), 'OLS: MSE on validation and test samples (CV)', '', 'polynomial degree', ['val','test'], range(1,poly_degree_max+1), True, savefig = savefigs, figname = 'OLS_val_test' + fignamePostFix)

# displaying heatmaps as functions of polynomial degree and regularization parameters
heatmap(MSE_cv_val_Ridge, 'Ridge validaton MSE (CV)', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max+1), True, savefig = savefigs, figname = 'Ridge_val_cv' + fignamePostFix)
heatmap(MSE_cv_test_Ridge, 'Ridge test MSE (CV)', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max+1), True, savefig = savefigs, figname = 'Ridge_test_cv' + fignamePostFix)
heatmap(MSE_test_Ridge, 'Ridge test MSE', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max+1), True, savefig = savefigs, figname = 'Ridge_test' + fignamePostFix)
heatmap(MSE_test_Lasso, 'Lasso test MSE', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max_Lasso+1), True, savefig = savefigs, figname = 'Lasso_test' + fignamePostFix)
heatmap(MSE_test_Lasso-MSE_test_Ridge[0:poly_degree_max_Lasso,:], 'Lasso-Ridge test MSE', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max_Lasso+1), True, savefig = savefigs, figname = 'Lasso_Ridge_test' + fignamePostFix)
heatmap(MSE_cv_val_Lasso, 'Lasso validaton MSE (CV)', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max_Lasso+1), True, savefig = savefigs, figname = 'Lasso_val_cv' + fignamePostFix)
heatmap(MSE_cv_test_Lasso, 'Lasso test MSE (CV)', '\u03BB', 'polynomial degree', lambda_tests_str, range(1,poly_degree_max_Lasso+1), True, savefig = savefigs, figname = 'Lasso_test_cv' + fignamePostFix)

# compare OLS analytical and scikitlearn
plot_several(np.repeat(poly_degrees[:,None], 2, axis=1), np.hstack((MSE[:,None],MSE_scikitlearn[:,None])), ['r-', 'b-'], ['Analytical implementation', 'Scikit-Learn'], 'Polynomial degree', 'MSE', 'MSE: Analytical implementation vs Scikit-Learn (OLS)', savefig = savefigs, figname = 'SelfImpVsScikit' + fignamePostFix)

# plot MSE vs lambda and polynomial degree
plot_several(np.repeat(np.log10(lambda_tests[:,None]), poly_degree_max, axis=1), MSE_cv_val_Ridge.T, ['r-', 'b-'], np.arange(poly_degree_max)+1, 'log(\u03BB)', 'MSE', 'Ridge: MSE validation dependence on \u03BB and polynomial degree (CV)', savefig = savefigs, figname = 'Ridge_val_cv_plot' + fignamePostFix)
plot_several(np.repeat(np.log10(lambda_tests[:,None]), poly_degree_max_Lasso, axis=1), MSE_cv_val_Lasso.T, ['r-', 'b-'], np.arange(poly_degree_max_Lasso)+1, 'log(\u03BB)', 'MSE', 'Lasso: MSE validation dependence on \u03BB and polynomial degree (CV)', savefig = savefigs, figname = 'Lasso_val_cv_plot' + fignamePostFix)
plot_several(np.repeat(np.log10(lambda_tests[:,None]), poly_degree_max, axis=1), MSE_test_Ridge.T, ['r-', 'b-'], np.arange(poly_degree_max)+1, 'log(\u03BB)', 'MSE', 'Ridge: MSE test dependence on \u03BB and polynomial degree', savefig = savefigs, figname = 'Ridge_test_plot' + fignamePostFix)
plot_several(np.repeat(np.log10(lambda_tests[:,None]), poly_degree_max_Lasso, axis=1), MSE_test_Lasso.T, ['r-', 'b-'], np.arange(poly_degree_max_Lasso)+1, 'log(\u03BB)', 'MSE', 'Lasso: MSE test dependence on \u03BB and polynomial degree', savefig = savefigs, figname = 'Lasso_test_plot' + fignamePostFix)

# plot train vs test error
plot_several(np.repeat(poly_degrees[:,None], 2, axis=1), np.hstack((MSE_train[:,None],MSE_test[:,None])), ['r-', 'b-'], ['Train', 'Test'], 'Polynomial degree', 'MSE', 'MSE: Train vs. test (OLS)', savefig = savefigs, figname = 'OLS_train_test' + fignamePostFix)

#plot bias and variance as function of polynomial degree
plot_several(np.repeat(poly_degrees[:,None], 3, axis=1), np.hstack((MSE_cv_test_OLS[:,None],bias_plus_noise_cv_test_OLS[:,None],variance_cv_test_OLS[:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', '', 'Bias-variance trade-off (CV, OLS)', savefig = savefigs, figname = 'bias_var_cv_OLS' + fignamePostFix)
plot_several(np.repeat(poly_degrees[:,None], 3, axis=1), np.hstack((MSE_bootstrap_OLS[:,None],bias_bootstrap_OLS[:,None],variance_bootstrap_OLS[:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', '', 'Bias-variance trade-off (Bootstrap, OLS)', savefig = savefigs, figname = 'bias_var_bootstrap_OLS' + fignamePostFix)
plot_several(np.repeat(poly_degrees[:,None], 3, axis=1), np.hstack((MSE_bootstrap_test_Ridge[:,8][:,None],bias_bootstrap_test_Ridge[:,8][:,None],variance_bootstrap_test_Ridge[:,8][:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', '', 'Bias-variance trade-off (Ridge, lambda=333)', savefig = savefigs, figname = 'bias_var_bootstrap_Ridge' + fignamePostFix)
plot_several(np.repeat(poly_degrees[0:poly_degree_max_Lasso,None], 3, axis=1), np.hstack((MSE_cv_test_Lasso[:,1][:,None],bias_cv_test_Lasso[:,1][:,None],variance_cv_test_Lasso[:,1][:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', '', 'Bias-variance trade-off (Lasso, lambda=something)', savefig = savefigs, figname = 'bias_var_cv_Lasso' + fignamePostFix)

pdb.set_trace()

# displaying OLS whole sample results
max_features = len(var_beta_list[-1])
for i in range(poly_degree_max):
    var_beta_list[i].extend([np.nan]*(max_features-len(var_beta_list[i])))
#var_beta_list = list(map(list, zip(*var_beta_list)))
var_beta_array = np.asarray(var_beta_list).T
conf_interval_beta = 1.96*np.sqrt(var_beta_array)
fig, axs = plt.subplots()
collabel = [('Degree ' + str(x+1)) for x in range(min(5,poly_degree_max))]
rowlabel = ['MSE','R2']
#axs.set_title('OLS results whole dataset (no test sample)')
table_list_of_features = list_of_features[min(5-1,poly_degree_max)]
conf_interval_beta_text = ['Confidence: ' + i for i in table_list_of_features]
rowlabel.extend(conf_interval_beta_text)
axs.axis('off')
the_table = axs.table(cellText=np.around(np.vstack((MSE[0:min(5,poly_degree_max)], R2[0:min(5,poly_degree_max)], conf_interval_beta[0:len(list_of_features[min(5-1,poly_degree_max)]),0:min(5,poly_degree_max)])),decimals = 3),colLabels=collabel,rowLabels = rowlabel, loc='center')
plt.savefig('TableOLSWhole' + fignamePostFix, dpi=300, bbox_inches='tight')
plt.show()

# Chart: show how Lasso sets certain betas to zero (pick degree 5)
# Chart: show bias variance trade-off for OLS with noise=0.2 and data separation 0.1
# or with noise=0.02 and data separation 0.1 and polynomials up to degree 11
# Chart: show lambda works with data sep 0.1 and sigma 0.1
# Chart: plot best model using surfplot
# Chart: plot best model difference using surfplot'
# Q: Are we supposed to get a proper bias-variance chart with reasonably low noise? I don't think so actually...
# Q: Are we supposed to get improvement with lambda with reasonaly low noise? I don't think so actually...
# Q: Lasso convergence problem!
# Seaborn table: need fewer lambdas or rotate axis title 