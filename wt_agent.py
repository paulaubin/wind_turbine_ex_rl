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

import itertools
from tqdm import tqdm

from rl_glue import RLGlue

from agents import BaseAgent
import tiles3 as tc


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
	print('')

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

	def agent_init(self, agent_info={}):
		print('')

	def agent_policy(self, active_tiles):
		print('')

	def agent_start(self, state):
		print('')

	def agent_step(self, reward, state):
		print('')

def run_experiment(environment, agent, agent_parameters, \
	experiment_parameters):
	print('')

