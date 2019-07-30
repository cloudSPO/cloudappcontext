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

class CostDataManager:
	def __init__(self, path, outcome_label, cost_labels, used_features, rows2drop=[None], data_mode="mrs"):        
		self.path = path
		self.mode = "cost"
		self.outcome_label = outcome_label
		self.rows2drop = rows2drop
		self.cost_labels = cost_labels
		self.used_features = used_features
		self.cost_label_to_index = {}
		for i in range(0, len(self.cost_labels)):
			self.cost_label_to_index[self.cost_labels[i]] = i
			print(i, self.cost_labels[i])
		xlsx = pd.ExcelFile(self.path)
		self.sheet_names = xlsx.sheet_names
		self.transform_x = self.__transformX
		self.transform_y = self.__transformY
		self.data_mode = data_mode

	def setXTransform(self, transform_x):
		self.transform_x = transform_x

	def setYTransform(self, transform_y):
		self.transform_y = transform_y
		
	def load(self):    
		# dictionary of sheet data -> 'openmrs_{small/medium/large}_database_{large/xlarge}'
		metrics_dict = {}
		for sheet_name in self.sheet_names:
			single_sheet = pd.read_excel(self.path, sheet_name=sheet_name, header=0)
			metrics_dict[sheet_name] = single_sheet
	
		# all  metrics combined
		self.all_metrics = pd.concat(metrics_dict.values(), sort=False)
		self.all_metrics.dropna(axis=1, how='all', inplace=True)
		
		self.metrics = self.all_metrics[:]
		
		# metrics for database large & xlarge
		metrics_large = []
		metrics_xlarge = []
		
		#metrics for video
		metrics_small_video = []
		metrics_medium_video = []
		metrics_large_video = []
		
		# split dataset based on database instance (large or xlarge)
		if self.data_mode != "video":
			for k,v in metrics_dict.items():
				metric_category_label = k.split("_")[-1]
				if metric_category_label == "large":
					metrics_large.append(v)
				if metric_category_label == "xlarge":
					metrics_xlarge.append(v)
					
			self.metrics_large = pd.concat(metrics_large, sort=False)
			self.metrics_large.dropna(axis=1, how='all', inplace=True)

			self.metrics_xlarge = pd.concat(metrics_xlarge, sort=False)
			self.metrics_xlarge.dropna(axis=1, how='all', inplace=True)
		
		else:
			# split dataset based on database instance (large or xlarge)
			for k,v in metrics_dict.items():
				metric_category_label = k.split("_")[-1]
				if metric_category_label == "large":
					metrics_large_video.append(v)
				if metric_category_label == "medium":
					metrics_medium_video.append(v)
				if metric_category_label == "small":
					metrics_small_video.append(v)
					
			self.metrics_large_video = pd.concat(metrics_large_video, sort=False)
			self.metrics_large_video.dropna(axis=1, how='all', inplace=True)

			self.metrics_medium_video = pd.concat(metrics_medium_video, sort=False)
			self.metrics_medium_video.dropna(axis=1, how='all', inplace=True)
			
			self.metrics_small_video = pd.concat(metrics_small_video, sort=False)
			self.metrics_small_video.dropna(axis=1, how='all', inplace=True)
		
		# remove unused features 
		keep = []
		keep.extend(self.cost_labels)
		keep.extend(self.used_features)
		if self.data_mode != "video":
			self.metrics_large = self.metrics_large.loc[:, self.metrics_large.columns.isin(keep)]
			self.metrics_xlarge = self.metrics_xlarge.loc[:, self.metrics_xlarge.columns.isin(keep)]
		else:
			self.metrics_large_video = self.metrics_large_video.loc[:, self.metrics_large_video.columns.isin(keep)]
			self.metrics_medium_video = self.metrics_medium_video.loc[:, self.metrics_medium_video.columns.isin(keep)]
			self.metrics_small_video = self.metrics_small_video.loc[:, self.metrics_small_video.columns.isin(keep)]
		self.metrics = self.metrics.loc[:, self.metrics.columns.isin(keep)]
		
		# 'flatten' the dataset on all cost categories
		self.metrics = self.__oneHotEncodingMinusOne(self.metrics, self.cost_labels)  
		if self.data_mode != "video":
			self.metrics_large = self.__oneHotEncodingMinusOne(self.metrics_large, self.cost_labels)
			self.metrics_xlarge = self.__oneHotEncodingMinusOne(self.metrics_xlarge, self.cost_labels)      
		else:
			self.metrics_large_video = self.__oneHotEncodingMinusOne(self.metrics_large_video, self.cost_labels)
			self.metrics_medium_video = self.__oneHotEncodingMinusOne(self.metrics_medium_video, self.cost_labels) 
			self.metrics_small_video = self.__oneHotEncodingMinusOne(self.metrics_small_video, self.cost_labels) 
			
		# remove outliers 1
		self.metrics = self.__removeOutliersByStd(self.metrics, "all")  
		if self.data_mode != "video":
			self.metrics_large = self.__removeOutliersByStd(self.metrics_large, "large")
			self.metrics_xlarge = self.__removeOutliersByStd(self.metrics_xlarge, "xlarge") 
		else:
			self.metrics_large_video = self.__removeOutliersByStd(self.metrics_large_video, "large")
			self.metrics_medium_video = self.__removeOutliersByStd(self.metrics_medium_video, "medium")
			self.metrics_small_video = self.__removeOutliersByStd(self.metrics_small_video, "small")
							
		# remove outliers
		self.metrics.reset_index(drop=True, inplace=True)
		self.metrics.drop(self.metrics.index[self.rows2drop], inplace=True, errors='ignore') 
		if self.data_mode != "video":
			self.metrics_large.reset_index(drop=True, inplace=True)
			self.metrics_large.drop(self.metrics_large.index[self.rows2drop], inplace=True, errors='ignore') 
			self.metrics_xlarge.reset_index(drop=True, inplace=True)
			self.metrics_xlarge.drop(self.metrics_xlarge.index[self.rows2drop], inplace=True, errors='ignore')
		else:
			self.metrics_large_video.reset_index(drop=True, inplace=True)
			self.metrics_large_video.drop(self.metrics_large_video.index[self.rows2drop], inplace=True, errors='ignore') 
			self.metrics_medium_video.reset_index(drop=True, inplace=True)
			self.metrics_medium_video.drop(self.metrics_medium_video.index[self.rows2drop], inplace=True, errors='ignore')
			self.metrics_small_video.reset_index(drop=True, inplace=True)
			self.metrics_small_video.drop(self.metrics_small_video.index[self.rows2drop], inplace=True, errors='ignore')
		
		print("\nCOST")
		print(self.metrics['cost'])
		print("\n")
		self.all_metrics.loc[len(self.all_metrics)]=self.metrics['cost']
	
	def getAllMetrics(self):
		return self.all_metrics
	
	def getMetrics(self):
		return self.metrics 
	
	def getMetricsDBLarge(self):
		if self.data_mode!="mrs":
			raise Exception('Data mode must be set to mrs for this method to work!')
		return self.metrics_large 
	
	def getMetricsDBXLarge(self):
		if self.data_mode!="mrs":
			raise Exception('Data mode must be set to mrs for this method to work!')
		return self.metrics_xlarge
	
	def getMetricsVideoDBLarge(self):
		return self.metrics_large_video 
	
	def getMetricsVideoDBMedium(self):
		return self.metrics_medium_video 
		
	def getMetricsVideoDBSmall(self):
		return self.metrics_small_video
	
	def getXY(self, metrics):
		X,Y =  self.getXY_panda(metrics)
		return X.values,Y.values
	
	def getXY_panda(self, metrics):
		X = metrics.loc[:, metrics.columns != self.outcome_label]
		X = X.fillna(X.mean()).astype('float')
		# set outcome variable
		Y = metrics[self.outcome_label]
		Y = Y.fillna(Y.mean()).astype('float')

		# make transformations    
		Y = self.transform_y(Y) 
		X = self.transform_x(X) 

		return X,Y
		
	def exportDataset(self):
		writer = pd.ExcelWriter(self.mode + '_dataset.xlsx', engine='xlsxwriter')
		self.metrics.to_excel(writer, sheet_name = "metrics")
		if self.data_mode != "video":
			self.metrics_large.to_excel(writer, sheet_name = "metrics_large")
			self.metrics_xlarge.to_excel(writer, sheet_name = "metrics_xlarge")
		else:
			self.metrics_large_video.to_excel(writer, sheet_name = "metrics_video_large")
			self.metrics_medium_video.to_excel(writer, sheet_name = "metrics_video_medium")
			self.metrics_small_video.to_excel(writer, sheet_name = "metrics_video_small")
		writer.save()
		writer.close()
	
	def __transformX(self, X):
		scaler = MinMaxScaler()
		X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)       
		return np.log(X+1)
	
	def __transformY(self, Y):
		data = Y.copy().values
		data = np.sqrt(data)
		ret = pd.Series(data)
		return ret 
	
	def __oneHotEncoding(self, metrics, cost_labels):
		features = metrics.loc[:, metrics.columns != self.outcome_label]
		result = []        

		cost_values =  []
		for j in range(0, len(cost_labels)):
			tmp = features.copy()
			for k in range(0, len(cost_labels)):
				if k!=j:
					tmp[cost_labels[k]] = 0.0
			tmp[cost_labels[j]] = 1.0
			tmp[self.outcome_label] = metrics[cost_labels[j]]
			result.append(tmp)
		dataset = pd.concat(result)

		return dataset
	
	
	def __oneHotEncodingMinusOne(self, metrics, cost_labels):
		features = metrics.loc[:, metrics.columns != self.outcome_label]
		result = []        

		cost_values =  []
		for j in range(0, len(cost_labels)):
			tmp = features.copy()
			for k in range(0, len(cost_labels)-1):
				tmp[cost_labels[k]] = 0.0
				if k != (len(cost_labels)-1):
					tmp[cost_labels[j]] = 1.0
			tmp.drop(labels = [cost_labels[-1]], axis = 1, inplace=True)
			tmp[self.outcome_label] = metrics[cost_labels[j]]
			result.append(tmp)
		dataset = pd.concat(result)        
		return dataset
	
	#keep only the ones that are within +3 to -3 standard deviations
	def __removeOutliersByStd(self, data, data_label):
		len1 = len(data)
		tmp = data.copy()
		for f in data.columns:
			tmp = tmp[np.abs(tmp[f]-tmp[f].mean())<=(3*tmp[f].std())]

		len2 = len(tmp)
		print(len1-len2,"features removed (std +/- 3) - ", data_label)
		return tmp             
		