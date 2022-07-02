# Implement a RL agent to maximize the production of a wind turbine

## Before you start
This repo was done for personnal use, the code could be simplified and respect some conding rules. Yet if you are interested in the project please feel free to send a message and I will do some clean up.

Some files were taken from external sources such as :
* Tile coding [http://incompleteideas.net/tiles/tiles3.py-remove]
* RlGlue [https://github.com/andnp/coursera-rl-glue]
* Coursera Notebook

## The wind turbine environment
It consists of a single wind turbine that gets wind as input (speed, direction) and that outputs power.

The wind is relative to the wind turbine, a 0deg angle means that the wind turbine is facing the wind. The power is maximum when facing the wind and no action is taken.

There are 3 actions that can be made :
* +1 = rotate clockwise
* -1 = rotate anticlockwise
* 0 = do nothing
When the wind turbine is rotating, the power output gets a penalty (it lowers)

The wind speed is constant but the wind heading experiences a slight random walk with time

[ADD PICTURE HERE]

There are 3 parameters that can be set on the environment :
* The relative wind heading at start
* The wind speed random walk rate
* The wind heading random walk rate

In the code you will see that the power could be filter to induce a delay due to the inertia of the rotor (same thing could apply to actions by the way). Yet this funcionnality is not used because I am not sure if it breaks the MDP framework.

## The reinforement learning agent
It is a softmax actor-critic as described in Reinforcement Learning: An Introduction, 2nd Edition [http://incompleteideas.net/book/RLbook2020.pdf] by Sutton and Bartho.

It learns how to derive a stochastic policy that tends to optimality on the given environment. An example can be found here averaged other 100 runs on 10 000 timesteps.