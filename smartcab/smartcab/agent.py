import random
import pdb
import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env,alpha,epsilon,gamma):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
	# TODO: Initialize any additional variables here
	self.succesfull_trials = 0 # keep track of success rate
	self.gamma = gamma # Discount Factor	
	self.alpha = alpha # Learning Rate
	self.epsilon = epsilon
	self.actions = [None,'left','right','forward']
	self.state_space = list(itertools.product(*[['green','red'],[True,False],[True,False],self.actions]))
	self.QMatrix = pd.DataFrame(np.zeros((len(self.state_space),4)),
	index= pd.MultiIndex.from_tuples(self.state_space,names=['light','Oncoming', 'Left','Waypoint']),columns=self.actions)	

    def get_current_state(self,inputs,waypoint):
	return (inputs['light'],not(inputs['oncoming'] in ['right','forward']),not(inputs['left'] == 'forward'),waypoint)				
	
    def updateQValue(self,old_state,action,new_state,reward):
	newQValue = reward + self.gamma*np.max(self.QMatrix.loc[new_state])
	self.QMatrix.loc[old_state][action] *= (1-self.alpha)
	self.QMatrix.loc[old_state][action] += self.alpha*newQValue

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
	self.state = self.get_current_state(inputs,self.next_waypoint)

        # Select action according to your policy
        rnd = random.uniform(0,1) 
	if rnd < self.epsilon:
		action = random.choice(self.actions)
	else:
		action = random.choice(self.QMatrix.columns[self.QMatrix.loc[self.state] == self.QMatrix.loc[self.state].max()])

        # Execute action and get reward
        reward = self.env.act(self, action)

	# Check if we have reached our destination
	if(reward > 2):
		self.succesfull_trials +=1

        # Learn policy based on state, action, reward
	new_state = self.get_current_state(self.env.sense(self),self.planner.next_waypoint())
	self.updateQValue(self.state,action,new_state,reward)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    results = []
    for gamma in [.85,.9,.95]:
	for alpha in [.01,.03,.1]:
		for epsilon in [.01,.03,.05]:	
		    e = Environment()  # create environment (also adds some dummy traffic)
		    a = e.create_agent(LearningAgent,alpha=alpha,epsilon=epsilon,gamma=gamma)  # create agent
		    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

		    # Now simulate it
		    sim = Simulator(e, update_delay=.01, display=False)  # create simulator (uses pygame when display=True, if available)

		    sim.run(n_trials=500)  # run for a specified number of trials
		    succes_rate = a.succesfull_trials*1./500 # Calculate succes rate
		    results.append((gamma,alpha,epsilon,succes_rate)) # Append to results
		    if(gamma == .95 and alpha == .03 and epsilon == .01): # print out Q-matrix of optimal parameter configuration
		    	with open("Q-Matrix.tex","w") as table_file:
				table_file.write(a.QMatrix.to_latex(na_rep='None'))		    						   
    print results

if __name__ == '__main__':
    run()
