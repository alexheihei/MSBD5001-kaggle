import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import calendar
import datetime
import numpy as np
import matplotlib.pylab as plt
import os.path as osp
import random
import time
import math
from statistics import mean, stdev
from copy import copy, deepcopy
from pprint import pprint

import pdb

class config:
	training_mode = False
	train_file = '../input/train.csv'
	test_file = '../input/test.csv'
	sample_submission = '../input/sampleSubmission.csv'
	checkpt = '../weights/parameters.txt'
	step = 0.001
	iteration = 2

def plotSpeed(train):
	plt.plot(train['timestamp'], train['speed'])
	plt.ylabel('Speed')
	plt.xlabel('timestamp')
	plt.show()

def getTimeStamp(date):
	date_split = date.split(' ')
	ts = time.mktime(datetime.datetime.strptime(date_split[0], "%d/%m/%Y").timetuple())
	if len(date_split) > 1:
		ts += int(date.split(' ')[1].split(':')[0]) * 60 * 60
	return int(ts) - 1483200000

holidays = ['1/1/2017', '28/1/2017', '29/1/2017', '30/1/2017', '31/1/2017', '4/4/2017', '14/4/2017', '15/4/2017', '17/4/2017', '3/5/2017', '1/5/2017', '30/5/2017', '1/7/2017', '1/10/2017', '5/10/2017', '28/10/2017', '25/12/2017',
	'1/1/2018', '16/2/2018', '17/2/2018', '18/2/2018', '19/2/2018', '30/3/2018', '31/3/2018', '2/4/2018', '5/4/2018', '1/5/2018', '22/5/2018', '18/6/2018', '1/7/2018', '25/9/2018', '1/10/2018', '17/10/2018', '25/12/2018']

# Typhoon timestamp
typhoon_ts = [(14047200, 14090400), (20239200, 20289600), (53848800, 53989200)]
typhoon_date = ['12/6/2017', '23/8/2017', '16/9/2018', '17/9/2018']


day2num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6, 'Holiday': 7}
def findDay(date):
	date = date.split(' ')[0]
	if date in holidays:
		return 7

	day = datetime.datetime.strptime(date, '%d/%m/%Y').weekday() 
	return day

train = pd.read_csv(config.train_file)
sample_df = pd.read_csv(config.sample_submission)

train['weekday'] = train.apply(lambda row: findDay(row.date), axis=1)
train['hour'] = train.apply(lambda row: int(row.date.split()[1].split(':')[0]), axis=1)
train['timestamp'] = train.apply(lambda row: getTimeStamp(row.date), axis=1)
only2017 = train[train.timestamp < 31536000]

test = pd.read_csv(config.test_file)
test['weekday'] = test.apply(lambda row: findDay(row.date), axis=1)
test['hour'] = test.apply(lambda row: int(row.date.split()[1].split(':')[0]), axis=1)
test['timestamp'] = test.apply(lambda row: getTimeStamp(row.date), axis=1)

meandict = {} # weekday and hour mean
stddict = {}
for d in range(8):
	for h in range(24):
		selected = train[(train.weekday == d) & (train.hour == h)].sort_values('speed')
		selected = selected[len(selected) // 40 : -len(selected) // 40] # ignore 5% extreme values
		meandict[(d, h)] = selected.speed.mean()
		stddict[(d, h)] = selected.speed.std()
		

def interpolation(row, train):
	rowMean = meandict[(int(row.weekday), int(row.hour))]

	i = 1
	while not (train.timestamp == row.timestamp - 3600 * i).any():
		if row.timestamp - 3600 * i < 0:
			return rowMean
		i += 1
	
	left = train[train.timestamp == row.timestamp - 3600 * i]
	leftMean = meandict[(int(left.weekday), int(left.hour))]
	leftSpeed = float(left.speed)

	j = 1
	while not (train.timestamp == row.timestamp + 3600 * j).any():
		if row.timestamp + 3600 * j > 365 * 2 * 24 * 3600:
			return rowMean
		j += 1

	right = train[train.timestamp == row.timestamp + 3600 * j]
	rightMean = meandict[(int(right.weekday), int(right.hour))]
	rightSpeed = float(right.speed)

	leftRatio = abs(rowMean - leftMean) / (abs(rowMean - leftMean) + abs(rowMean - rightMean))
	rightRatio = abs(rowMean - rightMean) / (abs(rowMean - leftMean) + abs(rowMean - rightMean))

	if rowMean >= max(leftMean, rightMean):
		return max(leftSpeed, rightSpeed)
	
	if rowMean <= min(leftMean, rightMean):
		return min(leftSpeed, rightSpeed)
	
	return leftSpeed * rightRatio + rightSpeed * leftRatio

def predict(row, train, meandict, k): # k is parameter
	# group number [1715, 1253, 237, 216, 67]
	for ts in typhoon_ts:
		if row.timestamp >= ts[0] and row.timestamp <= ts[1]:
			# typhoon is in force
			return interpolation(row, train)

	m = meandict[(row.weekday, row.hour)]
	meansd = mean([stddict[k] for k in stddict])

	base = meansd / stddict[(row.weekday, row.hour)]

	# Idea: trust the mean more if the std is small, otherwise rely on its neighbours' speed
	param = [[k[0] * (base ** k[1]), k[2] * (base ** k[3])],
			[k[4] * (base ** k[5]), k[6] * (base ** k[7])]] # should be large if std is small

	offset = 0
	count = 0
	if (train.timestamp == row.timestamp - 3600).any(): # 1 hour before
		before = train[train.timestamp == row.timestamp - 3600]
		offset += float(before.speed) - meandict[(int(before.weekday), int(before.hour))]
		count += 1
	
	if (train.timestamp == row.timestamp + 3600).any(): # 1 hour later
		after = train[train.timestamp == row.timestamp + 3600]
		offset += float(after.speed) - meandict[(int(after.weekday), int(after.hour))]
		count += 1
	
	if count > 0:
		offset /= param[count - 1][0]

	else:
		if (train.timestamp == row.timestamp - 7200).any(): # 2 hours before
			before = train[train.timestamp == row.timestamp - 7200]
			offset += float(before.speed) - meandict[(int(before.weekday), int(before.hour))]
			count += 1
		
		if (train.timestamp == row.timestamp + 7200).any(): # 2 hours later
			after = train[train.timestamp == row.timestamp + 7200]
			offset += float(after.speed) - meandict[(int(after.weekday), int(after.hour))]
			count += 1
		
		if count > 0:
			offset /= param[count - 1][1]
		
		
	return m + offset

def meansqscore(selected_train, remain_train, k):
	selected_train['pred_speed'] = selected_train.apply(lambda row: predict(row, remain_train, meandict, k), axis=1)
	selected_train['sqerr'] = selected_train.apply(lambda row: (row.pred_speed - row.speed) ** 2, axis=1)


	return selected_train['sqerr'].sum() / len(selected_train)


if config.training_mode:
	timestampChoice = np.array(train[train.timestamp >= 31536000].timestamp) - 31536000

	selected_train = only2017[~only2017.timestamp.isin(timestampChoice)]
	remain_train = only2017[only2017.timestamp.isin(timestampChoice)]

	for _ in range(config.iteration):
		k = np.genfromtxt(config.checkpt)
		print('Loaded param ', k)
		print('Err =', meansqscore(selected_train, remain_train, k))

		for i, v in enumerate(k):
			# init direction
			k[i] += config.step
			pos_err = meansqscore(selected_train, remain_train, k)
			k[i] -= config.step * 2
			neg_err = meansqscore(selected_train, remain_train, k)
			k[i] += config.step
			prev_err = meansqscore(selected_train, remain_train, k)
			if pos_err < neg_err:
				k_dir = 16
			else:
				k_dir = -16

			while True:
				k[i] += k_dir * config.step
				print('Set k[{}] to {:.3f}'.format(i, k[i]))
				post_err = meansqscore(selected_train, remain_train, k)
				print('New error =', post_err, end='\n\n')
				if post_err >= prev_err:
					k[i] -= k_dir * config.step
					if int(abs(k_dir)) <= 1:
						break
					else:
						k_dir = int(k_dir) // 2
						continue

				prev_err = post_err
		
		final_err = meansqscore(selected_train, remain_train, k)
		print('Final Err =', final_err)
		print('K =', k)
		#print(selected_train.sort_values(by=['sqerr'], ascending=False).head(10))
		np.savetxt(config.checkpt, k, delimiter=' ', fmt='%f')

else:
	k = np.genfromtxt(config.checkpt)
	test['speed'] = test.apply(lambda row: predict(row, train, meandict, k), axis=1)

	pred = test[['id', 'speed']]
	pred.to_csv('submission.csv', index=False)
