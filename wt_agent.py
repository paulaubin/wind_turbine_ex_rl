#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/06/04
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Wind turbine 2D optimisation exercice - agent code """
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import itertools
from tqdm import tqdm

from rl_glue import RLGlue

from agents import BaseAgent
import tiles3 as tc
import plot_script

from wind_turbine import WindTurbineEnvironment

'''
from typing import Iterable, Any
from itertools import product
'''

class WindTurbineTileCoder:
	def __init__(self, iht_size=4092, num_tilings=32, num_tiles=8):
		"""
		Initializes the WindTurbine Tile Coder
		Initializers:
		iht_size -- int, the size of the index hash table, 
		typically a power of 2

		num_tilings -- int, the number of tilings
		num_tiles -- int, the number of tiles.
		Here both the width and height of the tiles are the same

		Class Variables:
		self.iht -- tc.IHT, the index hash table that
		the tile coder will use
		self.num_tilings -- int, the number of tilings
		the tile coder will use
		self.num_tiles -- int, the number of tiles
		the tile coder will use
		"""
		self.num_tilings = num_tilings
		self.num_tiles = num_tiles
		self.iht = tc.IHT(iht_size)

	def get_tiles(self, wind_speed, wind_heading):
		"""
		Takes in a wind_speed and a wind_heading
		from the wind turbine environment and returns
		a numpy array of active tiles.
		
		Arguments:
		wind_speed -- float, the speed of the wind
		between 0 and infinity
		wind_heading -- float, the relative heading of the wind
		turbine between -pi and pi

		returns:
		tiles -- np.array, active tiles
		"""
		wind_speed_scaled = 1/np.exp(wind_speed/15.0) \
			* self.num_tiles
		wind_heading_scaled = wind_heading * np.pi/180 \
			* self.num_tiles

		tiles = tc.tileswrap(self.iht, self.num_tilings, \
			[wind_speed_scaled, wind_heading_scaled], \
			wrapwidths=[self.num_tiles, False])

		return np.array(tiles)

''' CHECK TILINGS INVESTIGATION SHOULD BE CONTINUED
speed = np.linspace(0, 20, num=20)
heading = np.linspace(-180, 180, num=36)
test_obs = list(itertools.product(speed, heading))
wttc = WindTurbineTileCoder(iht_size=4096, \
	num_tilings=2, num_tiles=2)
result=[]
for obs in test_obs:
	speed, heading = obs
	tiles = wttc.get_tiles(wind_speed=speed, wind_heading=heading)
	print('speed = ', repr(speed))
	print('heading = ', repr(heading))
	print('tiles = ', repr(tiles))
	result.append(tiles)
print('tiles = ', repr(tiles))
'''

def compute_softmax_prob(actor_w, tiles):
	"""
	Computes softmax probability for all actions
	It is defined as
	pi(a|s,theta) = exp(h(s,a,theta))/sum_b(exp(h(s,b,theta) - c)
	and normalized as
	pi(a|s,theta) 
		= exp(h(s,a,theta) - c)/sum_b(exp(h(s,b,theta) - c)
	where c = max_b(h(s,b,theta))
	
	Args:
	actor_w - np.array, an array of actor weights
	tiles - np.array, an array of active tiles
	
	Returns:
	softmax_prob - np.array, an array of size equal to \
	num. actions, and sums to 1.
	"""
	# Compute the list of state-action preferences
	state_action_preferences = [actor_w[a][tiles].sum() \
		for a in range(actor_w.shape[0])]

	# Set the constant c by finding the maximum of state-action
	# preferences
	c = np.max(state_action_preferences)

	# Compute the numerator by subtracting c from state-action
	# preferences and exponentiating it
	numerator = np.exp(state_action_preferences - c)

	# Compute the denominator by summing the values 
	# in the numerator
	denominator = np.sum(numerator)

	# Create a probability array by dividing each element \
	# in numerator array by denominator
	softmax_prob = numerator / denominator

	return softmax_prob

'''
# CHECK SOFTMAX
iht_size = 4096
num_tilings = 8
num_tiles = 8
test_tc = WindTurbineTileCoder(iht_size=iht_size, \
		num_tilings=num_tilings, num_tiles=num_tiles)
num_actions = 3
actions = list(range(num_actions))
actor_w = np.zeros((len(actions), iht_size))
# setting actor weights such that state-action
# preferences are always [-1, 1, 2]
actor_w[0] = -1./num_tilings
actor_w[1] = 1./num_tilings
actor_w[2] = 2./num_tilings
# obtain active_tiles from state
state = [0, -180]
speed, heading = state
active_tiles = test_tc.get_tiles(speed, heading)
# compute softmax probability
softmax_prob = compute_softmax_prob(actor_w, active_tiles)
print('softmax probability: {}'.format(softmax_prob))
assert np.allclose(softmax_prob, \
	[0.03511903, 0.25949646, 0.70538451])
'''

class ActorCriticSoftmaxAgent(BaseAgent):
	def __init__(self):
		self.rand_generator = None
		self.actor_step_size = None
		self.critic_step_size = None
		self.avg_reward_step_size = None
		self.tc = None
		self.avg_reward = None
		self.critic_w = None
		self.actor_w = None
		self.actions = None
		self.softmax_prob = None
		self.prev_tiles = None
		self.last_action = None
		self.step_count = None
		self.verbose = None

	def agent_init(self, agent_info={}):
		"""Setup for the agent called when the experiment
		first starts.

		Set parameters needed to setup the semi-gradient
		TD(0) state aggregation agent.

		Assume agent_info dict contains:
		{
			"iht_size": int
			"num_tilings": int,
			"num_tiles": int,
			"actor_step_size": float,
			"critic_step_size": float,
			"avg_reward_step_size": float,
			"num_actions": int,
			"seed": int
		}
		"""

		# set random seed for each run
		self.rand_generator \
			= np.random.RandomState(agent_info.get("seed")) 

		iht_size = agent_info.get("iht_size")
		num_tilings = agent_info.get("num_tilings")
		num_tiles = agent_info.get("num_tiles")

		# initialize self.tc to the tile coder we created
		self.tc = WindTurbineTileCoder(iht_size=iht_size, \
			num_tilings=num_tilings, num_tiles=num_tiles)

		# set step-size accordingly
		# (we normally divide actor and critic step-size
		# by num. tilings (p.217-218 of Sutton and Barto textbook))
		self.actor_step_size \
			= agent_info.get("actor_step_size")/num_tilings
		self.critic_step_size = \
			agent_info.get("critic_step_size")/num_tilings
		self.avg_reward_step_size = \
			agent_info.get("avg_reward_step_size")

		self.actions = list(range(agent_info.get("num_actions")))

		# Set initial values of average reward, actor weights,
		# and critic weights
		# We initialize actor weights to three times the iht_size. 
		# Recall this is because we need to have one set
		# of weights for each of the three actions.
		self.avg_reward = 0.0
		self.actor_w = np.zeros((len(self.actions), iht_size))
		self.critic_w = np.zeros(iht_size)

		self.softmax_prob = None
		self.prev_tiles = None
		self.last_action = None
		self.step_count = 0
		self.verbose = agent_info.get("verbose")
	
	def agent_policy(self, active_tiles):
		""" policy of the agent
		Args:
			active_tiles (Numpy array): active tiles returned 
			by tile coder
			
		Returns:
			The action selected according to the policy
		"""
		
		# compute softmax probability
		softmax_prob = compute_softmax_prob(self.actor_w, \
			active_tiles)
		
		# Sample action from the softmax probability array
		# self.rand_generator.choice() selects an element
		# from the array with the specified probability
		chosen_action = self.rand_generator.choice( \
			self.actions, p=softmax_prob)
		
		# save softmax_prob as it will be useful later when 
		# updating the Actor
		self.softmax_prob = softmax_prob
		
		return chosen_action

	def agent_start(self, state):
		"""The first method called when the experiment starts,
		called after the environment starts.
		Args:
			state (Numpy array): the state from the environment's
			env_start function.
		Returns:
			The first action the agent takes.
		"""

		speed, heading = state

		### Use self.tc to get active_tiles using speed and
		# heading (2 lines)
		# set current_action by calling self.agent_policy
		# with active_tiles
		
		active_tiles = self.tc.get_tiles(state[0], state[1])
		current_action = self.agent_policy(active_tiles)

		self.last_action = current_action
		self.prev_tiles = np.copy(active_tiles)

		return self.last_action


	def agent_step(self, reward, state):
		"""A step taken by the agent.
		Args:
			reward (float): the reward received for taking the
			last action taken
			state (Numpy array): the state from the environment's
								step based on 
								where the agent ended up after the
								last step.
		Returns:
			The action the agent is taking.
		"""
		if self.verbose:
			print('======================================')
			print('Entering Step ', repr(self.step_count))
			self.step_count += 1
			print('self.avg_reward_step_size = ', repr(self.avg_reward_step_size))
			print('self.critic_step_size = ', repr(self.critic_step_size))
			print('self.self.actor_step_size = ', repr(self.actor_step_size))

		speed, heading = state

		# Use self.tc to get active_tiles using speed and
		# heading
		active_tiles = self.tc.get_tiles(speed, heading)
		
		# Compute delta using Equation (1)
		delta = reward - self.avg_reward \
			+ self.critic_w[active_tiles].sum() \
			- self.critic_w[self.prev_tiles].sum()
		if self.verbose:
			print('------------------------------------')
			print('compute delta')
			print('delta = ', repr(delta))
			print('reward = ', repr(reward))
			print('self.avg_reward = ', repr(self.avg_reward))
			print('average reward diff = ', repr(reward - self.avg_reward))
			print('v(S[t+1], w) = ', repr(self.critic_w[active_tiles].sum()))
			print('v(S[t], w) = ', repr(self.critic_w[self.prev_tiles].sum()))

		# update average reward using Equation (2)
		self.avg_reward += self.avg_reward_step_size * delta
		if self.verbose:
			print('------------------------------------')
			print('increment average reward')
			print('avg_reward increment = ', repr(self.avg_reward_step_size * delta))
			print('self.avg_reward = ', repr(self.avg_reward))

		# update critic weights using Equation (3) and (5)
		self.critic_w[self.prev_tiles] \
			+= self.critic_step_size * delta
		if self.verbose:
			print('------------------------------------')
			print('update critic weights')
			print('weight increment = ', repr(self.critic_step_size * delta))
			print('self.critic_w[self.prev_tiles] = ', repr(self.critic_w[self.prev_tiles]))

		# update actor weights using Equation (4) and (6)
		# We use self.softmax_prob saved from the previous timestep
		for a in self.actions:
			if a == self.last_action:
				self.actor_w[a][self.prev_tiles] \
					+= self.actor_step_size * delta \
					* (1 - self.softmax_prob[a])
			else:
				self.actor_w[a][self.prev_tiles] \
					+= self.actor_step_size * delta \
					* (0 - self.softmax_prob[a])
			if self.verbose:
				print('------------------------------------')
				print('update actor weight')
				print('self.actor_w[', a ,'][self.prev_tiles] = ', \
					repr(self.actor_w[a][self.prev_tiles]))

		### set current_action by calling 
		# self.agent_policy with active_tiles
		current_action = self.agent_policy(active_tiles)
		if self.verbose:
			print('------------------------------------')
			print('choose new action')
			print('action = ', repr(current_action))

		self.prev_tiles = active_tiles
		self.last_action = current_action

		return self.last_action

	def agent_message(self, message):
		if message == 'get avg reward':
			return self.avg_reward


# Define function to run experiment
def run_experiment(environment, agent, environment_parameters, \
	agent_parameters, experiment_parameters):

	rl_glue = RLGlue(environment, agent)
			
	# sweep agent parameters
	for num_tilings in agent_parameters['num_tilings']:
		for num_tiles in agent_parameters["num_tiles"]:
			for actor_ss in agent_parameters["actor_step_size"]:
				for critic_ss in agent_parameters["critic_step_size"]:
					for avg_reward_ss in agent_parameters["avg_reward_step_size"]:
						
						env_info = environment_parameters
						agent_info = {"num_tilings": num_tilings,
									  "num_tiles": num_tiles,
									  "actor_step_size": actor_ss,
									  "critic_step_size": critic_ss,
									  "avg_reward_step_size": avg_reward_ss,
									  "num_actions": agent_parameters["num_actions"],
									  "iht_size": agent_parameters["iht_size"],
									  "verbose": agent_parameters["verbose"]}

						policy_test_wind_speed_range = range(5,16)			
						policy_test_wind_heading_range = range(-180,181)

						# results to save
						return_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
						exp_avg_reward_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
						wind_heading_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
						action_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
						policy_distrib = pd.DataFrame(data = {'ws' : [ws for ws in policy_test_wind_speed_range for wh in policy_test_wind_heading_range], \
							'wh' : [wh for ws in policy_test_wind_speed_range for wh in policy_test_wind_heading_range]})
						
						# used to calculte the mean and the std iteratively 
						#policy_sum = {ws : {wh : np.zeros(agent_info["num_actions"]) \
						#	for wh in policy_distrib["wind_heading_range"]} \
						#	for ws in policy_distrib["wind_speed_range"]}
						# used to calculate the std iteratively
						#policy_sum_sq = {ws : {wh : np.zeros(agent_info["num_actions"]) \
						#	for wh in policy_distrib["wind_heading_range"]} \
						#	for ws in policy_distrib["wind_speed_range"]}
						#policy_distrib['policy_sum'] = list(np.zeros((len(policy_distrib), agent_info["num_actions"])))
						#policy_distrib['policy_sum_sq'] = list(np.zeros((len(policy_distrib), agent_info["num_actions"])))
						run_count = 0
						policy_sum = np.zeros((len(policy_distrib), agent_info["num_actions"]))
						policy_sum_sq = np.zeros((len(policy_distrib), agent_info["num_actions"]))
						#print('policy_distrib = ', repr(policy_distrib))


						# using tqdm we visualize progress bars 
						for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):
							env_info["seed"] = run
							agent_info["seed"] = run
				
							rl_glue.rl_init(agent_info, env_info)
							rl_glue.rl_start()

							num_steps = 0
							total_return = 0.
							return_arr = []
							wind_heading = 0. # MAYBE FALSE, DEPENDS ON INIT
							action_log = 1

							# exponential average reward without initial bias
							exp_avg_reward = 0.0
							exp_avg_reward_ss = 0.01
							exp_avg_reward_normalizer = 0

							while num_steps < experiment_parameters['max_steps']:
								num_steps += 1
								
								rl_step_result = rl_glue.rl_step()
								
								reward, state, action, _ = rl_step_result
								wind_heading = state[1]

								total_return += reward
								return_arr.append(reward)
								avg_reward = rl_glue.rl_agent_message("get avg reward")

								exp_avg_reward_normalizer = exp_avg_reward_normalizer + exp_avg_reward_ss * (1 - exp_avg_reward_normalizer)
								ss = exp_avg_reward_ss / exp_avg_reward_normalizer
								exp_avg_reward += ss * (reward - exp_avg_reward)
								
								return_per_step[run-1][num_steps-1] = total_return
								exp_avg_reward_per_step[run-1][num_steps-1] = exp_avg_reward
								wind_heading_per_step[run-1][num_steps-1] = wind_heading
								action_per_step[run-1][num_steps-1] = action
							
							run_count += 1
							for i in range(len(policy_distrib)):
								ws = policy_distrib['ws'][i]
								wh = policy_distrib['wh'][i]
								cur_policy = np.array(get_policy_distribution(rl_glue.agent, [ws, wh]))
								policy_sum[i] += cur_policy
								policy_sum_sq[i] += cur_policy**2
								#print('ws = ', repr(ws))
								#print('wh = ', repr(wh))
								#print('cur_policy = ', repr(cur_policy))
								#print('policy_sum = ', repr(policy_sum))
								#print('policy_sum_sq = ', repr(policy_sum_sq))
						
						policy_distrib['action_proba_avg'] = list(policy_sum/run_count)
						policy_distrib['action_proba_std'] = list(np.sqrt(policy_sum_sq/run_count \
							- (policy_sum/run_count)**2))
						#print('policy_distrib = ', repr(policy_distrib))


						if not os.path.exists('results'):
							os.makedirs('results')
				
						save_name = "ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}"\
							.format(num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)
						total_return_filename = "results/{}_total_return.npy".format(save_name)
						exp_avg_reward_filename = "results/{}_exp_avg_reward.npy".format(save_name)
						wind_heading_filename = "results/{}_wind_heading.npy".format(save_name)
						action_filename = "results/{}_action.npy".format(save_name)
						policy_distrib_filename = "results/{}_policy_distrib.npy".format(save_name)

						np.save(total_return_filename, return_per_step)
						np.save(exp_avg_reward_filename, exp_avg_reward_per_step)
						np.save(wind_heading_filename, wind_heading_per_step)
						np.save(action_filename, action_per_step)
						policy_distrib.to_pickle(policy_distrib_filename)

						n_max = experiment_parameters['max_steps']
						score_range = range(int(np.floor(0.9*n_max)), \
							int(np.floor(n_max)))
						score = np.mean(exp_avg_reward_per_step[:, score_range])
						print('score = ', repr(score))


'''
def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
	for params in product(*parameters.values()):
		yield dict(zip(parameters.keys(), params))
'''

def get_policy_distribution(agent, state):
	"""Gets the policy probabilistic distribution of the agent for a given state
		Args:
			agent (rlagent): a reinforcement learning agent such as
			the softmax actor critic agent
			state (Numpy array): the current state of the agent
		Returns:
			The policy probabilistic distribution
	"""
	wind_speed, wind_heading = state
	active_tiles = agent.tc.get_tiles(wind_speed, wind_heading)
	softmax_prob = compute_softmax_prob(agent.actor_w, active_tiles)

	return softmax_prob


#### Run Experiment
np.random.seed(100)

# Experiment parameters
experiment_parameters = {
	"max_steps" : 20000, #20000,
	"num_runs" : 10, #50
}

# Environment parameters
environment_parameters = {
	"random_angle_start" : True,
	"random_speed_start" : False,
	"speed_start" : 10,
	"angle_start" : 0,
	"wind_heading_var" : 0.1,
	"wind_speed_var" : 0.1,
}

# Agent parameters
# Each element is an array because we will be later sweeping
# over multiple values actor and critic step-sizes
# are divided by num. tilings inside the agent
agent_parameters = {
	"num_tilings": [32],
	"num_tiles": [8],
	"actor_step_size": [2**(-1)], #[2**(-2)],
	"critic_step_size": [2**(-1)], #[2**1],
	"avg_reward_step_size": [2**(-0)], #[2**(-6)],
	"num_actions": 3,
	"iht_size": 16384, #4096,
	"verbose" : False
}
'''
param_grid = {
	"actor_step_size" : agent_parameters["actor_step_size"]
	"critic_step_size": agent_parameters["critic_step_size"]
	"avg_reward_step_size" : agent_parameters["avg_reward_step_size"]
}
'''

current_env = WindTurbineEnvironment
current_agent = ActorCriticSoftmaxAgent

run_experiment(current_env, current_agent, environment_parameters, \
	agent_parameters, experiment_parameters)
plot_script.plot_result(agent_parameters, 'results')

