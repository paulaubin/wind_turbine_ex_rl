# Implement a RL agent to maximize the production of a wind turbine

## Disclaimer
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

Below is an illusrtation of how the wind turbine environment works

![Example of wind turbine in action](https://github.com/paulaubin/wind_turbine_ex_rl/blob/master/plot/environment_example.png)

No strategy is implemented here, the action taken are only to get a feeling of how it works

There are 3 parameters that can be set on the environment :
* The relative wind heading at start
* The wind speed random walk rate
* The wind heading random walk rate

In the code you will see that the power could be filter to induce a delay due to the inertia of the rotor (same thing could apply to actions by the way). Yet this funcionnality is not used because I am not sure if it breaks the MDP framework.

## The reinforement learning agent
It is an average reward softmax actor-critic as described in Reinforcement Learning: An Introduction, 2nd Edition [http://incompleteideas.net/book/RLbook2020.pdf] by Sutton and Bartho.

It learns how to derive a stochastic policy that tends to optimality on the given environment. An illustration can be found below, where the agent was trained over 10 000 timesteps and averaged over 100 episods and with random relative angle at start

## Training on a case without penalty for action that rotate the turbine

On the figure below, the agent was trained on an environment that does not penalises action. The figure matches our intuition where it should head to 0 angle for any offset angle. The training steps where 10 000 and the episods where 100 with random relative angle at start


![Policy example after training](https://github.com/paulaubin/wind_turbine_ex_rl/blob/master/plot/ang_rand_whv0p1_wsp0p1_step10k_run100_score_0p0028.png)

## Training with penalty for action that rotate the turbine

Here a penalty of 1% of the turbine maximum power is applied at every time step that the turbine rotates.

We can also have a look at the full policy over wind speed and wind heading on the graph below

![3D Policy example with action penalty](https://github.com/paulaubin/wind_turbine_ex_rl/blob/master/plot/3d_plot/embed_test.html)
{% include index.html %}

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://paulaubin.github.io/wind_turbine_ex_rl/" height="525" width="100%"></iframe>