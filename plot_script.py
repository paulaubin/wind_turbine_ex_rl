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

def plot_result(agent_parameters, results_folder):
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
	policy_distrib = np.load(policy_distrib_filename)

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
	policy_angle = policy_distrib[:,:,0]
	policy_distrib_action_trigo = policy_distrib[:,:,1]
	policy_distrib_action_do_nothing = policy_distrib[:,:,2]
	policy_distrib_action_clockwise = policy_distrib[:,:,3]
	mean_policy_angle = np.mean(policy_angle, 0)
	mean_policy_distrib_action_trigo = \
		np.mean(policy_distrib_action_trigo, 0)
	mean_policy_distrib_action_do_nothing = \
		np.mean(policy_distrib_action_do_nothing, 0)
	mean_policy_distrib_action_clockwise = \
		np.mean(policy_distrib_action_clockwise, 0)
	std_policy_distrib_action_trigo = \
		np.std(policy_distrib_action_trigo, 0)
	std_policy_distrib_action_do_nothing = \
		np.std(policy_distrib_action_do_nothing, 0)
	std_policy_distrib_action_clockwise = \
		np.std(policy_distrib_action_clockwise, 0)

	ax1 = plt.subplot(3, 1, 1)
	plt.scatter(mean_policy_angle, mean_policy_distrib_action_trigo, label='rotate trigo')
	plt.scatter(mean_policy_angle, mean_policy_distrib_action_do_nothing, label='do nothing')
	plt.scatter(mean_policy_angle, mean_policy_distrib_action_clockwise, label='clockwise')
	plt.fill_between(mean_policy_angle, \
		mean_policy_distrib_action_trigo - std_policy_distrib_action_trigo, \
		mean_policy_distrib_action_trigo + std_policy_distrib_action_trigo,
		alpha = .1)
	plt.fill_between(mean_policy_angle, \
		mean_policy_distrib_action_do_nothing - std_policy_distrib_action_do_nothing, \
		mean_policy_distrib_action_do_nothing + std_policy_distrib_action_do_nothing,
		alpha = .1)

	plt.fill_between(mean_policy_angle, \
		mean_policy_distrib_action_clockwise - std_policy_distrib_action_clockwise, \
		mean_policy_distrib_action_clockwise + std_policy_distrib_action_clockwise,
		alpha = .1)
	
	plt.xlabel('Angle (deg)')
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

	plt.show()


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
