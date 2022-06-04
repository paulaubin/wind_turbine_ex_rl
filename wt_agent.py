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
		print('')

	def get_tiles(self, wind_speed, wind_heading):
		print('')

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

