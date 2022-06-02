#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/05/29
# version ='1.0'
# header reference : https://www.delftstack.com/howto/python/common-header-python/
# ---------------------------------------------------------------------------
""" Wind turbine 2D optimision exercice """
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from butterworth_low_pass_filter_template \
	import butter_lowpass, butter_lowpass_filter, manual_filter
from scipy.signal import lfilter, lfilter_zi

# Define the wind turbine
@dataclass
class wind_turbine:
	"""
	A wind turbine in 2D from the source
	https://www.raeng.org.uk/publications/other/23-wind-turbine
	"""
	cp = 0.4
	s = np.pi*50**2 											# m2
	rho = 1.225 												# kg.m-3

	rotor_cutoff = 1/30										# Hz
	filter_order = 4
	b, a = butter_lowpass(rotor_cutoff, 1.0, 4)
	zi = lfilter_zi(b, a)
	angle_increment = 0.1 									# deg
	control_cost = 1e-1										# MW
	control_on = False

	# HIDE VARIABLES
	wind_sp_hist = np.array(np.zeros(filter_order+1))			# m.s-1
	wind_rel_heading_hist = np.array(np.zeros(filter_order+1))# deg
	power_hist_filt = np.array(np.zeros(filter_order+1))  	# MW
	power_hist = np.array(np.zeros(filter_order+1))  			# MW
	data_counter = 0
	wind_sp = 0													# m.s-1
	wind_rel = 0												# deg
	power_balance = 0 											# MW


	def update_filter_wind_history(self):
		self.wind_sp_hist, _ = \
			lfilter(self.b, self.a, self.wind_sp_hist, \
				zi=self.zi*self.wind_sp_hist_filt[0:-1])

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

		power_hist = 1/1e6*np.cos(np.pi/180.0 \
			*self.wind_rel_heading_hist)*1/2 \
			*self.cp*self.rho*self.s*self.wind_sp_hist**3

		self.power_hist = power_hist

		# Filter the given power to simulate the rotor inertia
		if self.data_counter <= self.filter_order:
			self.power_hist_filt = np.array(np.mean( \
				power_hist[-self.data_counter:]) \
				*np.ones(len(power_hist)))
		else:
			power_hist_filt = manual_filter(self.b, self.a, \
				self.power_hist, self.power_hist_filt[:-1])
			self.power_hist_filt[-1] = power_hist_filt[-1]

		# Apply penalty due to control
		if self.control_on :
			self.power_balance = self.power_hist_filt[-1] \
				- self.control_cost
		else:
			self.power_balance = self.power_hist_filt[-1]


		'''
		if self.power_hist_filt[0] == 0:
			self.power_hist_filt = np.array(power_hist[-1] \
				*np.ones(np.size(self.power_hist_filt)))
		else:
			power_hist_filt, _ = lfilter(self.b, self.a, \
				power_hist, zi=self.zi*self.power_hist_filt[0])
			self.power_hist_filt[-1] = power_hist_filt[-1]

		if self.control_on == True :
			self.power_hist[-1] -= self.control_cost
			#self.power_hist_filt[-1] -= self.control_cost
		#print('self.power_hist_filt = ', repr(self.power_hist_filt))
		#print('self.power_hist = ', repr(self.power_hist))
		#self.power_hist_filt[2] = self.power_hist_filt[1] \
		#	+ 1/50 * (power_hist[2] - self.power_hist_filt[1])
		'''

	def get_wind(self, wind_speed, differential_wind_heading):
		for i in range(1,self.filter_order+1):
			self.wind_sp_hist[i-1] = self.wind_sp_hist[i]
			self.wind_rel_heading_hist[i-1] = self.wind_rel_heading_hist[i]
		self.wind_sp_hist[-1] = wind_speed
		self.wind_rel_heading_hist[-1] = self.wind_rel_heading_hist[-1] \
			+ differential_wind_heading
		self.data_counter += 1

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


wt = wind_turbine()
print('power output = ', wt.update_power_output())

fs = 1.0
T = 500.0         # value taken in seconds
n = int(T * fs) # indicates total samples
t = np.linspace(0, T, n, endpoint=False)

########### FAIRE UNE CLASSE WIND ###############
wind_speed = 10.0 + 1.0*np.sin(2*np.pi*0.2*t)
wind_heading = 0.0 + 1.0*10.0*np.sin(2*np.pi*0.3*t)
power_output_filt_log = np.array(np.zeros(np.size(t)))
power_output_log = np.array(np.zeros(np.size(t)))
power_control_log = np.array(np.zeros(np.size(t)))
power_balance_log = np.array(np.zeros(np.size(t)))
wind_speed_log = np.array(np.zeros(np.size(t)))
wind_heading_log = np.array(np.zeros(np.size(t)))
for w in range(np.size(t)):
#for w in range(7):
	if w > 100 and w <= 200:
		wt.rotate(+1)
	if w > 200 and w <= 250:
		wt.rotate(-1)
	if w > 250:
		wt.rotate(0)
	wt.get_wind(wind_speed[w], wind_heading[w])
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
wind_speed = 0.0 + np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
filtered_wind = np.array(np.zeros(np.size(wind)))
for w in range(np.size(wind)):
#for w in range(4):
	print('w = ', repr(w))
	wt.get_wind(wind[w])
	filtered_wind[w] = wt.filter_wind_history[2]

plt.plot(t, wind, 'b-', label='wind')
plt.plot(t, filtered_wind, 'g-', linewidth=2, label='filtered wind')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.show()
'''