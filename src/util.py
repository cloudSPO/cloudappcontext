import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from sklearn import metrics as m
import os
import math
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn import preprocessing
from scipy import stats
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot

class Util():
	def __init__(self):
		self.regression_results = pd.DataFrame()	
		self.inv_trans = lambda x: x ** 2
		
	def setInvTrans(self, inv_trans):
		self.inv_trans = inv_trans

	def visualize_prediction_error(self, model_info):
		model = model_info['model']
		X_train = model_info['X_train']
		X_test = model_info['X_test']
		Y_train = model_info['Y_train']
		Y_test = model_info['Y_test']

		visualizer = PredictionError(model)

		visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
		visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
		
		#visualizer.poof()			  # Draw/show/poof the data
		
        
        

	def visualize_residuals_plot(self, model_info):
		model = model_info['model']	   
		X_train = model_info['X_train']
		X_test = model_info['X_test']
		Y_train = model_info['Y_train']
		Y_test = model_info['Y_test']

		visualizer = ResidualsPlot(model)

		visualizer.fit(X_train, Y_train)  # Fit the training data to the model
		visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
		visualizer.poof()				  # Draw/show/poof the data
		
	def inverse_transform(self, pred, true, f):
		pred2 = np.array([f(i) for i in pred])
		true2 = np.array([f(i) for i in true])
		return pred2, true2
    
	def mean_absolute_percentage_error(self, true, pred):
		mask= true != 0
		mape= (np.fabs(true[mask] - pred[mask])/true[mask]).mean()        
		return mape*100
		
	def get_aic(self, pred, true, features_len):
		resid = true-pred
		sse = sum(resid**2)
		aic = 2*features_len - 2*math.log(sse)
		return aic
	
	def get_aic_c(self, pred, true, features_len):
		aic = self.get_aic(pred, true, features_len)
		num_observations = len(pred)
		aic_c = aic + ( 2*(features_len**2) + 2*features_len ) / (num_observations-features_len-1)
		return aic_c
		
	def get_bic(self, pred, true, features_len):
		resid = true-pred
		sse = sum(resid**2)
		num_observations = len(pred)
		bic = ( num_observations*math.log(sse/num_observations) ) + ( features_len*math.log(num_observations) )
		return bic
	
	def regression_report(self, true, pred, method_title, features_len):  
		info = pd.DataFrame()
		# calculate MAE
		pred, true = self.inverse_transform(pred,true,self.inv_trans)
		mae = m.mean_absolute_error(true, pred)
		print("MAE: " + str(mae))
		info['MAE'] = pd.Series(mae)

		# calculate MSE using scikit-learn
		mse = m.mean_squared_error(true, pred)
		print("MSE: " + str(mse))
		info['MSE'] = mse
        
        # calculate MAPE (Ivana)
		mape = self.mean_absolute_percentage_error(true, pred)
		print("MAPE: " + str(mape))
		info['MAPE'] = mape

		# calculate RMSE using scikit-learn
		rmse = np.sqrt(m.mean_squared_error(true, pred))
		print("RMSE: " + str(rmse))
		info['RMSE'] = rmse

		#calculate explained variance scores using scikit-learn
		evs = m.explained_variance_score(true, pred)
		print("EVS: " + str(evs))
		info['EVS'] = evs
		
		#calculate AIC
		aic = self.get_aic(pred, true, features_len)
		print("AIC: " + str(aic))
		info['AIC'] = aic
		
		#calculate AICc
		aic_c = self.get_aic_c(pred, true, features_len)
		print("AICc: " + str(aic_c))
		info['AICc'] = aic_c
		
		#calculate BIC
		bic = self.get_bic(pred, true, features_len)
		print("BIC: " + str(bic))
		info['BIC'] = bic

		#calculate r2 score using scikit-learn
		r2 = m.r2_score(true, pred)
		print("R2: " + str(r2))
		info['R2'] = r2

		#calculate r2(adj) score using scikit-learn
		r2 = m.r2_score(true, pred)
		r2_adj =  1 - (1-r2)*(len(true)-1)/(len(true)-features_len-1)
		print("R2(adj): " + str(r2_adj))
		info['R2(adj)'] = r2_adj
		
		#init aic weight and aic delta
		info['AIC delta'] = 0.0
		info['BIC delta'] = 0.0
		info['AIC weight'] = 0.0

		# save regression result in dataframe
		info['method'] = method_title
		info.set_index('method')
		self.regression_results = self.regression_results.append(info)
		
	#https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118856406.app5
	def get_aic_bic(self):
		info = pd.DataFrame()
		reg_res_aic = map(abs, self.regression_results['AIC'])
		reg_res_bic = map(abs, self.regression_results['BIC'])
		
		min_aic = min(reg_res_aic)
		min_bic = min(reg_res_bic)
		
		for i in range(0, len(self.regression_results)):
			diff_aic = abs(min_aic-abs(self.regression_results['AIC'][i]))
			self.regression_results['AIC delta'][i] = diff_aic
			
			diff_bic = abs(min_bic-abs(self.regression_results['BIC'][i]))
			self.regression_results['BIC delta'][i] = diff_bic
			
		aic_delta_sum = sum([math.exp(-0.5*i) for i in self.regression_results['AIC delta']])
		for i in range(0, len(self.regression_results)):
			w_m = math.exp(-0.5*self.regression_results['AIC delta'][i])/aic_delta_sum
			self.regression_results['AIC weight'][i] = w_m
			
	def get_r2(self, true, pred):
		return m.r2_score(true, pred)

	def get_standard_residuals(self, predicted, Y):
		Y = Y.values
		tmp = predicted-Y
		res_sum = sum(tmp**2)
		n = len(Y)
		div = np.sqrt(res_sum/(n-1))
		return (tmp*1.0) / div

	def get_internally_studentized_residuals(self, predicted, Y):
		ind = Y.index
		Y = Y.values
		residuals = [(Y[i] - predicted[i]) for i in range(len(Y))]
		Var_e = sum([((Y[i] - predicted[i])**2) for i in range(len(Y)) ]) / (len(Y) -2)
		SE_regression = Var_e**0.5
		studentized_residuals = [residuals[i]/SE_regression for i in range(len(residuals))] 
		return pd.Series(studentized_residuals, index=ind)

	def pred_true_graph(self, model_info, title): 
		Y_test = model_info['Y_test']
		pred = model_info['predicted']

		plt.scatter(Y_test,pred,c=None, s=40, alpha=0.75)
		plt.xlabel('Observed', fontsize = 'xx-large')
		plt.ylabel('Predicted', fontsize = 'xx-large')
		plt.tick_params(axis='both', labelsize=16)
		plt.title(title)
		plt.show()

	def residuals_graph(self, model_info, title):	 
		model = model_info['model']	   
		X_train = model_info['X_train']
		X_test = model_info['X_test']
		Y_train = model_info['Y_train']
		Y_test = model_info['Y_test']

		plt.scatter(model.predict(X_train), self.get_standard_residuals(model.predict(X_train),Y_train), c='b', s=40, alpha=0.5)
		plt.scatter(model.predict(X_test), self.get_standard_residuals(model.predict(X_test),Y_test), c='g', s=40, alpha=0.5)
		plt.hlines(y=0, xmin=0, xmax=50)
		plt.title(str(model))
		plt.ylabel('Standard Residuals')
		plt.xlabel('Fitted values')	   
		plt.title(title)
		plt.show()

	def qq_graph(self, model_info, title, pdf):
		model = model_info['model']	   
		X_train = model_info['X_train']
		X_test = model_info['X_test']
		Y_train = model_info['Y_train']
		Y_test = model_info['Y_test']

		fig, ax = plt.subplots(nrows=1)
		qqplot = sm.qqplot(self.get_standard_residuals(model.predict(X_test), Y_test), line='s', ax=ax)
		plt.title(title)
		pdf.savefig(fig)
		plt.show(qqplot)

	def scale_location_graph(self, model_info, title):
		model = model_info['model']	   
		X_train = model_info['X_train']
		X_test = model_info['X_test']
		Y_train = model_info['Y_train']
		Y_test = model_info['Y_test']

		scalelocplot = plt.plot(model.predict(X_test), abs(self.get_standard_residuals(model.predict(X_test), Y_test))**.5,	 'o')
		plt.xlabel('Fitted values')
		plt.ylabel('Square Root of |standardized residuals|')
		plt.title(title)
		plt.show(scalelocplot)

	def cook_distance_graph(self, model_info, title, pdf):
		fig, ax = plt.subplots(nrows=1)

		model = model_info['model']	   
		X_test = model_info['X_test']
		Y_test = model_info['Y_test']
		X = X_test	  

		model_leverage	= (X.T * np.linalg.inv(X.T.dot(X)).dot(X.T)).sum(0)
		model_norm_residuals = self.get_internally_studentized_residuals(model.predict(X_test), Y_test)

		model_leverage.sort_index(inplace=True)
		model_norm_residuals.sort_index(inplace=True)

		plt.scatter(model_leverage, model_norm_residuals, alpha=0.5, color='blue')
		#print(model_leverage.values)
		#print(model_norm_residuals.values)
		sns.regplot(model_leverage.values, model_norm_residuals.values, 
					scatter=False, 
					ci=False, 
					lowess=True,
					line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, 
				   ax = ax)
		for i, txt in enumerate(X.index):
			ax.annotate(txt, xy=(model_leverage.iloc[i], 
									   model_norm_residuals.iloc[i]))
		ax.set_title('Residuals vs Leverage ' + title)
		ax.set_xlabel('Leverage')
		ax.set_ylabel('Standardized Residuals')
		plt.show(fig)
		pdf.savefig(fig)

	def get_leverage(self, X):	  
		model_leverage	= (X.T * np.linalg.inv(X.T.dot(X)).dot(X.T)).sum(0)
		return model_leverage

	def get_cooks_distance(self, model_info, x_column_dimension):
		sr = self.get_standard_residuals(model_info['predicted'], model_info['Y_test'])
		lv = self.get_leverage(model_info['X_test'])
		cooks_distance = sr**2/((lv**2)*x_column_dimension)
		return cooks_distance
