# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pickle
import json
from example_1_full import get_subdataset, get_features, get_samples

def get_dataset(subject = 1,session = 1):
	sr = 200
	ch_fs_instances = []
	ch_tags_instances = []
	s_s_chs = get_subdataset(subject,session)
	_index = [i+1 for i,d in enumerate(s_s_chs[:,-1]) if d==1]
	instances = get_samples(_index, s_s_chs,sr)
	for f_instance in range(1,20): # len(instances) 60 instances
		ch_fs_i = []
		instance = np.array(instances[f_instance,:,1:-1]).transpose()
		ch_fs_instances.append(get_features(instance))
		ch_tags_instances.append('subject_{0}'.format(subject))
	return {"data": ch_fs_instances, "target": ch_tags_instances}


def eval_model(dataset, clf):
	false_accepted = 0
	Ok_accepted = 0
	total_tags = len(dataset['target'])
	for i,unk_entry in enumerate(dataset['target']):
		true_tag = dataset['target'][i]
		feature_vector = np.array([dataset['data'][i]])
		prediction = clf.predict(feature_vector)[0]
		accuracy = max(max(clf.predict_proba(feature_vector)))
		result_ = "True label: {0},  prediction: {1}, accuracy: {2}".format(true_tag, prediction, accuracy)
		print(result_)
		if true_tag == prediction:
			Ok_accepted+=1
		else:
			false_accepted+=1
	print('Ok_accepted {0}'.format(Ok_accepted))
	print('false_accepted {0}'.format(false_accepted))
	print('accuracy of Ok_accepted {0}'.format(round(Ok_accepted/total_tags,10)))
	print('accuracy of false_accepted {0}'.format(round(false_accepted/total_tags,10)))

subject = 12
session = 1
dataset = get_dataset(subject,session)
model = open('clf.sav', 'rb')
clf = pickle.load(model)
eval_model(dataset, clf)


