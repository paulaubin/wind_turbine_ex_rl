#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/06/06
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Wind turbine 2D optimisation exercice - plot helper
to comply with https://github.com/LucasBoTang/Coursera_Reinforcement_Learning/blob/master/03Prediction_and_Control_with_Function_Approximation/04Average_Reward_Softmax_Actor-Critic.ipynb """
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def plot_result(agent_parameters, results_folder, figure_counter):
	plt.rcParams.update({'figure.max_open_warning': 0})

	avg_reward_filename = results_folder + '/' + \
		get_pre_filename(agent_parameters) \
		+ '_exp_avg_reward.npy'
	total_return_filename = results_folder + '/' + \
		get_pre_filename(agent_parameters) \
		+ '_total_return.npy'
	wind_heading_filename = results_folder + '/' + \
		get_pre_filename(agent_parameters) \
		+ '_wind_heading.npy'
	action_filename = results_folder + '/' + \
		get_pre_filename(agent_parameters) \
		+ '_action.npy'
	policy_distrib_filename = results_folder + '/' + \
		get_pre_filename(agent_parameters) \
		+ '_policy_distrib.npy'

	avg_reward_data = np.load(avg_reward_filename)
	total_return_data = np.load(total_return_filename)
	wind_heading = np.load(wind_heading_filename)
	action = np.load(action_filename)
	policy_distrib = pd.read_pickle(policy_distrib_filename)
	#print('policy_distrib = ', repr(policy_distrib))

	label = 'actor_ss = ' \
		+ str(agent_parameters['actor_step_size']).strip('[]') \
		+ ' | critic_ss = ' \
		+ str(agent_parameters['critic_step_size']).strip('[]') \
		+ ' | avg_reward_ss = ' \
		+ str(agent_parameters['avg_reward_step_size']).strip('[]')

	mean_total_return_data = np.mean(total_return_data, 0)
	std_total_return_data = np.std(total_return_data, 0)
	mean_avg_reward_data = np.mean(avg_reward_data, 0)
	std_avg_reward_data = np.std(avg_reward_data, 0)
	mean_wind_heading = np.mean(wind_heading, 0)
	std_wind_heading = np.std(wind_heading, 0)
	mean_action = np.mean(action, 0)
	last_wind_heading = wind_heading[-1]
	first_wind_heading = wind_heading[0]

	policy_distrib['trigo_avg'] = policy_distrib.action_proba_avg.apply(lambda x : x[0])
	policy_distrib['nothing_avg'] = policy_distrib.action_proba_avg.apply(lambda x : x[1])
	policy_distrib['clockwise_avg'] = policy_distrib.action_proba_avg.apply(lambda x : x[2])

	policy_distrib['trigo_std'] = policy_distrib.action_proba_std.apply(lambda x : x[0])
	policy_distrib['nothing_std'] = policy_distrib.action_proba_std.apply(lambda x : x[1])
	policy_distrib['clockwise_std'] = policy_distrib.action_proba_std.apply(lambda x : x[2])
	
	'''fig = plt.figure()
	ax = Axes3D(fig)
	surf_trigo = ax.plot_trisurf(policy_distrib.ws, policy_distrib.wh, policy_distrib.trigo_avg, alpha = 0.5)
	surf_nothing = ax.plot_trisurf(policy_distrib.ws, policy_distrib.wh, policy_distrib.nothing_avg, alpha = 0.5)
	surf_clockwise = ax.plot_trisurf(policy_distrib.ws, policy_distrib.wh, policy_distrib.clockwise_avg, alpha = 0.5)
	ax.set_xlabel('Wind speed (m/s)', fontweight='bold')
	ax.set_ylabel('Wind heading (º)', fontweight='bold')
	ax.set_zlabel('Action probability', fontweight='bold')
	plt.show()'''
	'''
	wsu = policy_distrib['ws'].unique()
	whu = policy_distrib['wh'].unique()
	pol_trigo_avg = policy_distrib.trigo_avg.to_numpy().reshape(len(wsu), len(whu))
	pol_nothing_avg = policy_distrib.nothing_avg.to_numpy().reshape(len(wsu), len(whu))
	pol_clockwise_avg = policy_distrib.clockwise_avg.to_numpy().reshape(len(wsu), len(whu))
	fig = go.Figure(data=[go.Surface(x = wsu, y = whu, z = pol_trigo_avg.T, opacity = 1, name='rotate trigo', \
							cmin = 0, cmax = 1, colorscale = [[0, 'rgb(236, 243, 248)'], [1, 'rgb(54, 108, 150)']]),
						go.Surface(x = wsu, y = whu, z = pol_nothing_avg.T, opacity = 1, name='do nothing', \
							cmin = 0, cmax = 1, colorscale = [[0, 'rgb(253, 243, 231)'], [1, 'rgb(240, 131, 15)']]),
						go.Surface(x = wsu, y = whu, z = pol_clockwise_avg.T, opacity = 1, name='rotate clockwise', \
							cmin = 0, cmax = 1, colorscale = [[0, 'rgb(239, 247, 237)'], [1, 'rgb(74, 142, 62)']])])
	fig.update_layout(title='Action probability map')
	fig.update_layout(scene = dict(
                    xaxis_title='wind speed (m/s)',
                    yaxis_title='wind heading (º)',
                    zaxis_title='action probability',
					yaxis = dict(tickvals= list(np.linspace(-180, 180, 37))),
					zaxis = dict(range= [0,1])))
	pio.write_html(fig, file='index.html', auto_open=True)
	'''
	#fig.show()

	fig2 = plt.figure()	
	ax1 = plt.subplot(3, 1, 1)
	# Slice plot
	ws_slice = 10
	plt.scatter(policy_distrib[policy_distrib['ws']==ws_slice]['wh'], policy_distrib[policy_distrib['ws']==ws_slice]['trigo_avg'], label='rotate trigo')
	plt.scatter(policy_distrib[policy_distrib['ws']==ws_slice]['wh'], policy_distrib[policy_distrib['ws']==ws_slice]['nothing_avg'], label='do nothing')
	plt.scatter(policy_distrib[policy_distrib['ws']==ws_slice]['wh'], policy_distrib[policy_distrib['ws']==ws_slice]['clockwise_avg'], label='clockwise')
	plt.fill_between(policy_distrib[policy_distrib['ws']==ws_slice]['wh'],
		policy_distrib[policy_distrib['ws']==ws_slice]['trigo_avg'] - policy_distrib[policy_distrib['ws']==ws_slice]['trigo_std'], \
		policy_distrib[policy_distrib['ws']==ws_slice]['trigo_avg'] + policy_distrib[policy_distrib['ws']==ws_slice]['trigo_std'],
		alpha = .1)
	plt.fill_between(policy_distrib[policy_distrib['ws']==ws_slice]['wh'], \
		policy_distrib[policy_distrib['ws']==ws_slice]['nothing_avg'] - policy_distrib[policy_distrib['ws']==ws_slice]['nothing_std'], \
		policy_distrib[policy_distrib['ws']==ws_slice]['nothing_avg'] + policy_distrib[policy_distrib['ws']==ws_slice]['nothing_std'],
		alpha = .1)

	plt.fill_between(policy_distrib[policy_distrib['ws']==ws_slice]['wh'], \
		policy_distrib[policy_distrib['ws']==ws_slice]['clockwise_avg'] - policy_distrib[policy_distrib['ws']==ws_slice]['clockwise_std'], \
		policy_distrib[policy_distrib['ws']==ws_slice]['clockwise_avg'] + policy_distrib[policy_distrib['ws']==ws_slice]['clockwise_std'],
		alpha = .1)
	
	plt.xlabel('Angle (deg)')
	plt.ylim([0, 1])
	plt.ylabel('Action probability')
	plt.legend()
	plt.grid()

	ax2 = plt.subplot(3, 1, 2)
	plt.plot(mean_avg_reward_data, label=label)
	plt.fill_between(range(len(mean_avg_reward_data)), \
		mean_avg_reward_data - std_avg_reward_data,\
		mean_avg_reward_data + std_avg_reward_data,
		alpha=.1 )
	plt.xlabel('Training Steps')
	plt.ylabel('Exponential Average Reward')
	plt.legend()
	plt.grid()

	ax3 = plt.subplot(3, 1, 3, sharex=ax2)
	plt.plot(mean_wind_heading, label='mean wind heading')
	plt.fill_between(range(len(mean_wind_heading)), \
		mean_wind_heading - std_wind_heading,\
		mean_wind_heading + std_wind_heading,
		alpha=.1 )
	#for i in range(len(wind_heading)):
	#	plt.plot(wind_heading[i], '--') #, \
			#label='wind heading ' + str(i))
	#plt.plot(10 * (mean_action - 1), label='10x mean action')
	plt.xlabel('Training Steps')
	plt.ylabel('Wind heading and action')
	plt.legend()
	plt.grid()

	fig = plt.gcf()
	fig.set_size_inches(12, 8)
	plt.savefig('gif/result_' + str(figure_counter) + ' ' + label + '.png')
	plt.pause(0.1)


def get_pre_filename(agent_parameters):
	pre_filename = 'ActorCriticSoftmax' \
		+ '_tilings_' \
		+ str(agent_parameters['num_tilings']).strip('[]') \
		+ '_tiledim_' \
		+ str(agent_parameters['num_tiles']).strip('[]') \
		+ '_actor_ss_' \
		+ str(agent_parameters['actor_step_size']).strip('[]') \
		+ '_critic_ss_' \
		+ str(agent_parameters['critic_step_size']).strip('[]') \
		+ '_avg_reward_ss_' \
		+ str(agent_parameters['avg_reward_step_size']).strip('[]')
	return pre_filename
