#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/05/29
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Wind turbine 2D optimisation exercice - simulator code """
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from butterworth_low_pass_filter_template \
	import butter_lowpass, butter_lowpass_filter, manual_filter
from scipy.signal import lfilter, lfilter_zi
#import rlglue_environment
from rlglue_environment import BaseEnvironment


# Define the wind turbine
### TODO : SAME METHODS AS FOR
### https://github.com/andnp/coursera-rl-glue/blob/master/RLGlue/environment.py
@dataclass
class wind_turbine:
	"""
	A wind turbine in 2D from the source
	https://www.raeng.org.uk/publications/other/23-wind-turbine
	"""
	cp = 0.4
	s = np.pi*50**2 											# m2
	rho = 1.225 												# kg.m-3

	rotor_cutoff = 1/60											# Hz
	filter_order = 4
	b, a = butter_lowpass(rotor_cutoff, 1.0, filter_order)
	zi = lfilter_zi(b, a)
	angle_increment = 1.0 										# deg
	control_cost = 1e-1											# MW
	control_on = False

	wind_sp = 0													# m.s-1
	wind_rel = -30 												# deg
	wind_sp_hist = np.array(wind_sp * np.ones(filter_order+1))	# m.s-1
	wind_rel_heading_hist = np.array(wind_rel * np.ones(filter_order+1))# deg
	power_hist_filt = np.array(np.zeros(filter_order+1))  		# MW
	power_hist = np.array(np.zeros(filter_order+1))  			# MW
	max_power_hist = np.array(np.zeros(filter_order+1))  		# MW
	data_counter = 0
	power_balance = 0 											# MW
	max_power_balance = 0 										# MW

	def update_power_output(self):
		'''
		The instant power is calculated from 1/2.rho.a.cp.v^3
		with a projection from the wind angle.
		It uses a low-passed filter to take into account
		the inertia of the rotor
		'''
		# Get the instantaneous power
		for i in range(1,self.filter_order+1):
			self.power_hist_filt[i-1] = self.power_hist_filt[i]

		max_power_hist = 1/1e6*1/2 \
			*self.cp*self.rho*self.s*self.wind_sp_hist**3

		power_hist = np.abs(np.cos(np.pi/180.0 \
			*self.wind_rel_heading_hist)) * max_power_hist

		self.power_hist = power_hist
		self.max_power_hist = max_power_hist

		# Filter the given power to simulate the rotor inertia
		if self.data_counter <= self.filter_order:
			self.power_hist_filt = np.array(np.mean( \
				power_hist[-self.data_counter:]) \
				*np.ones(len(power_hist)))
			self.max_power_hist_filt = np.array(np.mean( \
				max_power_hist[-self.data_counter:]) \
				*np.ones(len(max_power_hist)))

		else:
			power_hist_filt = manual_filter(self.b, self.a, \
				self.power_hist, self.power_hist_filt[:-1])
			self.power_hist_filt[-1] = power_hist_filt[-1]
			max_power_hist_filt = manual_filter(self.b, self.a, \
				self.max_power_hist, self.max_power_hist_filt[:-1])
			self.max_power_hist_filt[-1] = max_power_hist_filt[-1]

		# Apply penalty due to control
		if self.control_on :
			# power balance could use the power filtered value
			# but not sure if markov
			self.power_balance = self.power_hist[-1] \
				- self.control_cost
			self.max_power_balance = self.max_power_hist[-1] \
				- self.control_cost # same
		else:
			self.power_balance = self.power_hist[-1]
			self.max_power_balance = self.max_power_hist[-1]


	def get_wind(self, wind_speed, differential_wind_heading):
		for i in range(1,self.filter_order+1):
			self.wind_sp_hist[i-1] = self.wind_sp_hist[i]
			self.wind_rel_heading_hist[i-1] = self.wind_rel_heading_hist[i]
		self.wind_sp_hist[-1] = wind_speed
		self.wind_rel_heading_hist[-1] += differential_wind_heading
		self.wind_rel_heading_hist = self.wind_rel_heading_hist + 180 % 360 - 180
		self.data_counter += 1

		self.wind_sp = self.wind_sp_hist[-1]
		self.wind_rel = self.wind_rel_heading_hist[-1]

	def rotate(self, direction):
		if direction == -1 :
			self.wind_rel_heading_hist[-1] += self.angle_increment
			self.control_on = True
		if direction == +1 :
			self.wind_rel_heading_hist[-1] -= self.angle_increment
			self.control_on = True
		if direction == 0 :
			# stays in place
			self.control_on = False
		if direction != -1 and direction != +1 and direction != 0 :
			print('wind turbine command ', direction, ' not valid')
			self.control_on = False

@dataclass
class wind:
	__speed_rate_mean = 0 			# m.s-1
	__speed_rate_std  = 0.0 		# m.s-1
	__heading_rate_mean = 0 		# deg
	__heading_rate_std  = 0.1 		# deg
	__time_step = 1 				# s
	__seed = 10 		# the random seed to repeat the results
	speed : float 					# m.s-1
	heading : float 				# deg
	__speed_init : float 			# m.s-1
	__heading_init : float 			# deg

	def __init__(self, speed=None, heading=None):
		self.speed = 10 if speed==None else wind_speed
		self.__speed_init = self.speed
		self.heading = 180 if heading==None else heading
		self.__heading_init = self.heading
		random.seed(self.__seed)

	def generate_wind(self):
		if self.__speed_rate_std != 0:
			speed_increment = np.random.normal(self.__speed_rate_mean + \
			(self.__speed_init - self.speed)/(10/self.__speed_rate_std), \
			self.__speed_rate_std, 1)
		else:
			speed_increment = np.random.normal(self.__speed_rate_mean + \
			(self.__speed_init - self.speed), \
			self.__speed_rate_std, 1)
		self.speed += speed_increment[0]/self.__time_step
		self.speed = np.maximum(0.0, self.speed)
		
		heading_increment = np.random.normal(self.__heading_rate_mean, \
			self.__heading_rate_std, 1)
		self.heading += heading_increment[0]/self.__time_step
		self.heading = self.heading % 360

@dataclass
class simu:
	__wind = wind()
	__wt = wind_turbine()
	__max_steps = float('inf')
	steps = 0
	state = {'wind_speed' :__wt.wind_sp, \
		'wind_rel_heading' : __wt.wind_rel, \
		'is_terminal' : False}
	reward : float
	__wind_heading_log = __wind.heading

	def __init__(self):
		self.reward = self.compute_reward

	def reset(self): # need to check that this is a proper reset
		self.__wind = wind()
		self.__wt = wind_turbine()
		self.steps = 0
		self.state = {'wind_speed' :self.__wt.wind_sp, \
			'wind_rel_heading' : self.__wt.wind_rel, \
			'is_terminal' : False}
		self.reward = self.compute_reward()
		self.__wind_heading_log = self.__wind.heading

	def step(self, action):
		self.steps += 1

		# Build the reward
		self.__wt.rotate(action)
		self.__wt.update_power_output()

		# Generate the next state
		# Iterate the wind
		self.__wind.generate_wind()
		wind_heading_diff = self.__wind.heading \
			- self.__wind_heading_log
		self.__wind_heading_log = self.__wind.heading

		# Iterate the power generated
		self.__wt.get_wind(self.__wind.speed, wind_heading_diff)

		# Log the data
		is_terminal = True if self.steps>=self.__max_steps else False
		self.state = {'wind_speed' :self.__wt.wind_sp, \
			'wind_rel_heading' : self.__wt.wind_rel, \
			'is_terminal' : is_terminal}
		self.reward = self.compute_reward()

	def compute_reward(self):
		reward = self.__wt.power_balance - self.__wt.max_power_balance
		if np.abs(self.__wt.wind_rel) > 90:
			reward = - (1 + (np.abs(self.__wt.wind_rel) - 90)/90) \
				* (self.__wt.max_power_balance \
				+ self.__wt.control_cost)
		return reward


	def get_power_balance(self):
		return self.__wt.power_balance

	def get_max_power_balance(self):
		return self.__wt.max_power_balance


class WindTurbineEnvironment(BaseEnvironment):
	__simu = simu()

	def __init__(self):
		reward = None
		observation = None
		termination = None
		self.reward_obs_term = (reward, observation, termination)

	def env_init(self, env_info={}):
		"""Setup for the environment called when the experiment
		first starts.
		Note:
			Initialize a tuple with the reward, first state
			observation, boolean
			indicating if it's terminal.
		"""
		self.__simu.reset()
		self.reward_obs_term \
			= (self.__simu.reward, [self.__simu.state['wind_speed'], \
				self.__simu.state['wind_rel_heading']], \
				self.__simu.state['is_terminal'])


	def env_start(self):
		"""The first method called when the experiment starts,
		called before the agent starts.

		Returns:
			The first state observation from the environment.
		"""
		obs = [self.__simu.state['wind_speed'], \
			self.__simu.state['wind_rel_heading']]
		return obs


	def env_step(self, action=0):
		"""A step taken by the environment.
		Args:
			action: The action taken by the agent

		Returns:
			(float, state, Boolean): a tuple of the reward,
			state observation,
				and boolean indicating if it's terminal.
		"""
		self.__simu.step(action - 1)
		self.reward_obs_term \
			= (self.__simu.reward, [self.__simu.state['wind_speed'], \
				self.__simu.state['wind_rel_heading']], \
				self.__simu.state['is_terminal'])

		return self.reward_obs_term


	def env_cleanup(self):
		"""Cleanup done after the environment ends"""
		self.__simu.reset()


	#def env_message(self, message):
		"""A message asking the environment for information
		Args:
			message: the message passed to the environment
		Returns:
			the response (or answer) to the message
		"""

'''
### Test simu class ###
sm = simu()
sm.reset()
wind_speed = np.array([])
wind_heading = np.array([])
power = np.array([])
max_power = np.array([])
reward = np.array([])
counter = 0
action = 0

while sm.state['is_terminal'] == False and counter < 600 :
	counter += 1
	#print('counter = ', repr(counter))
	#print('sm.state[wind_speed] = ', repr(sm.state['wind_speed']))
	if counter > 200:
		action = -1
	if counter > 500:
		action = +1
	if counter > 500:
		action = 0
	sm.step(action)
	wind_speed = np.append(wind_speed, [sm.state['wind_speed']])
	wind_heading = np.append(wind_heading, \
		sm.state['wind_rel_heading'])
	reward = np.append(reward, sm.reward)
	power = np.append(power, sm.get_power_balance())
	max_powre = np.append(max_power, sm.get_max_power_balance())


t = np.arange(len(wind_speed))
ax1 = plt.subplot(2, 1, 1)
plt.plot(t, wind_speed, label='wind speed from wt [m/s]')
plt.plot(t, wind_heading, label='wind relative heading from wt [deg]')
plt.xlabel('Time [sec]')
plt.tick_params('x', labelbottom=False)
plt.legend()
plt.grid()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(t, power, label='power')
plt.plot(t, reward, label='reward')
plt.xlabel('Time [sec]')
plt.ylabel('Power [MW]')
plt.legend()
plt.grid()
plt.show()
'''

### Test wind_turbine and wind class ###
'''
wt = wind_turbine()
wd = wind()

t = np.arange(600)
power_output_filt_log = np.array(np.zeros(np.size(t)))
power_output_log = np.array(np.zeros(np.size(t)))
power_control_log = np.array(np.zeros(np.size(t)))
power_balance_log = np.array(np.zeros(np.size(t)))
wind_speed_log = np.array(np.zeros(np.size(t)))
wind_heading_log = np.array(np.zeros(np.size(t)))
wind_heading_prev = wd.heading
for w in range(np.size(t)):
#for w in range(7):
	wd.generate_wind()
	wind_speed = wd.speed
	wind_heading = wd.heading
	wind_heading_diff = wd.heading - wind_heading_prev
	wind_heading_prev = wind_heading
	if w > 100 and w <= 200:
		wt.rotate(+1)
	if w > 200 and w <= 250:
		wt.rotate(-1)
	if w > 250:
		wt.rotate(0)
	wt.get_wind(wind_speed, wind_heading_diff)
	power_output = wt.update_power_output()
	power_output_filt_log[w] = wt.power_hist_filt[-1]
	power_output_log[w] = wt.power_hist[-1]
	power_balance_log[w] = wt.power_balance
	power_control_log[w] = wt.control_cost if wt.control_on else 0.0
	wind_speed_log[w] = wt.wind_sp_hist[-1]
	wind_heading_log[w] = wt.wind_rel_heading_hist[-1]

ax1 = plt.subplot(2, 1, 1)
#plt.plot(t, wind_speed, , '--', label='wind speed [m/s]')
plt.plot(t, wind_speed_log, label='wind speed from wt [m/s]')
#plt.plot(t, wind_heading, '--', label='relative wind heading [deg]')
plt.plot(t, wind_heading_log, label='wind relative heading from wt [deg]')
plt.xlabel('Time [sec]')
plt.tick_params('x', labelbottom=False)
plt.legend()
plt.grid()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(t, power_output_log, label='power')
plt.plot(t, power_output_filt_log, label='power_filt')
plt.plot(t, power_control_log, label='power lost to control the turbine')
plt.plot(t, power_balance_log, label='power balance')
plt.xlabel('Time [sec]')
plt.ylabel('Power [MW]')
plt.legend()
plt.grid()
plt.show()
'''