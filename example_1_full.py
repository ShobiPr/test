# -*- coding: utf-8 -*-
from __future__ import division
from pyhht import EMD
import numpy as np
import pylab as plt
import pickle
import json
import math
from scipy.stats import kurtosis
from scipy.stats import skew
import statistics as stats
import logging
from sklearn import svm
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

def SVM(dataset, C_F_V=10):
	clfArray = []
	meanScore = []
	_scores = []
	KERNELS = ['linear', 'rbf', 'sigmoid']
	for kernel in KERNELS:
		clf = svm.SVC(kernel=kernel, probability=True).fit(dataset['data'], dataset['target'])
		scores = cross_val_score(clf, dataset['data'], dataset['target'], cv=C_F_V)
		_scores.append(scores)
		meanScore.append(scores.mean())
		clfArray.append(clf)
	maxScore = max(meanScore)
	position = meanScore.index(maxScore)
	bestKernel = KERNELS[position]
	bestClf = clfArray[position]
	_std = np.std(_scores[position])
	return {"classifier": "{0} SVM".format(bestKernel), "accuracy": str(maxScore), "std": _std, "clf": bestClf}

def teager_energy(data):
	sum_values = sum(abs(data[x]**2) if x == 0
					 else abs(data[x]**2 - data[x - 1] * data[x + 1])
					 for x in range(0, len(data) - 1))
	return np.log10((1 / len(data)) * sum_values)

def instantaneous_energy(data):
	return np.log10((1 / len(data)) * sum(i ** 2 for i in data))

def hfd(a, k_max=None):
	L = []
	x = []
	N = a.size
	if not k_max:
		k_max = 10
	for k in range(1,k_max):
		Lk = 0
		for m in range(0,k):
			idxs = np.arange(1,int(np.floor((N-m)/k)), dtype=np.int32)
			Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
			Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
			Lk += Lmk
		L.append(float(np.log(Lk/(m+1))))
		x.append([float(np.log(1/ k)), 1])
	(p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
	return p[0]

def pfd(a):
	diff = np.diff(a)
	# x[i] * x[i-1] for i in t0 -> tmax
	prod = diff[1:-1] * diff[0:-2]
	# Number of sign changes in derivative of the signal
	N_delta = np.sum(prod < 0)
	n = len(a)
	return np.log(n)/(np.log(n)+np.log(n/(n+0.4*N_delta)))

def get_values_f(_vector):
	feat = []
	for ii,_vec in enumerate(_vector):
		feat += [
		# stats.mean(_vec),
		# np.var(_vec),
		# np.std(_vec),
		# np.median(_vec),
		# min(_vec),
		# max(_vec),
		# sum(_vec),
		# skew(_vec),
		# kurtosis(_vec),
		instantaneous_energy(_vec),
		teager_energy(_vec),
		hfd(_vec),
		pfd(_vec),
		]
	return feat

def get_imfs(signal):
	try:
		# decomposer_signal = EMD(signal, fixe=100, n_imfs=2)
		decomposer_signal = EMD(signal)
		imfs = decomposer_signal.decompose()
		if len(imfs) < 2:
			print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
			raise ValueError("imfs {}".format(len(imfs)))
		return imfs[:2]
	except Exception as e:
		raise e

def get_subdataset( _S=1, Sess=1):
	_file = 'train/Data_S%02d_Sess%02d.csv'%(_S,Sess)
	_f = open(_file).readlines()
	channels = []
	for i,_rows in enumerate(_f):
		if i>0:
			channels.append(eval(_rows))
		else:
			_headers = _rows
	return np.array(channels)

def get_samples(_index, s_s_chs, sr,_size=1.3):
	instances = []
	for _ind in _index:
		instances.append(s_s_chs[_ind:int(math.ceil(_ind+(_size*sr)))][:])
	return np.array(instances)

def get_features(instance):
	features_vector = []
	for i, channel in enumerate(instance):
		if i<10:
			imfs = get_imfs(channel)
			features_vector += get_values_f(imfs)
	return features_vector

def get_dataset():
	sr = 200
	ch_fs_instances = []
	ch_tags_instances = []
	# subjects_name = ["Luis", "Shobi", "Julie"]
	for subject in range(1, 3): # 2
		for session in range(1, 2): # 1
			s_s_chs = get_subdataset(subject,session)
			_index = [i+1 for i, d in enumerate(s_s_chs[:,-1]) if d==1]
			instances = get_samples(_index, s_s_chs,sr)
			for f_instance in range(1, 3): # len(instances) 2 instances
				ch_fs_i = []
				instance = np.array(instances[f_instance,:,1:-1]).transpose()
				ch_fs_instances.append(get_features(instance))
				ch_tags_instances.append('subject_{0}'.format(subject))
				# ch_tags_instances.append('{0}'.format(subjects_name[subject-1]))
	return {"data": ch_fs_instances, "target": ch_tags_instances}

if __name__ == '__main__':
	dataset = get_dataset()

	result = SVM(dataset,C_F_V=2)
	a = "{0}, accuracy {1} +- {2}".format(result['classifier'],result['accuracy'], result['std'])

	logging.basicConfig(filename='fest.log',
						filemode='a',
							level=logging.DEBUG)
	logging.info(a)
	print("aaaaa")
	# saving the model
	# model_name = 'clf_P300.sav'
	# pickle.dump(result["clf"], open(model_name, 'wb'))

