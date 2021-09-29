# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:09:41 2021

@author: Navid
"""
import numpy as np
from operator import itemgetter, attrgetter
from numpy import random,sqrt
import os
import sys
from numpy import linalg as la
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
from datetime import datetime
#from q_learning_bins import plot_running_avg, plot_cost_to_go
from cvxopt import matrix,solvers
import tensorflow_probability as tfp
import statsmodels.stats.correlation_tools as cv

'''blood objects'''

class blood:   
    
    def __init__(self,shelf_life):
        self.shelf_life = shelf_life
        
    def update_shelf_life(self):
        self.shelf_life = self.shelf_life -1
        
    def get_shelf_life(self):
        return self.shelf_life
    
    def __str__(self):
        return "blood bag with {} remaining life".format(self.shelf_life)
 
'''hospitals'''    
       
class hospital:   
    
    def __init__(self ,shortage_cost ,wastage_cost ,dist , distparams ):
        self.dist = dist
        self.distparams = distparams
        self.shortage_cost = shortage_cost
        self.wastage_cost = wastage_cost
        self.inventory = []
        self.shortage_penalty = 0
        self.wastage_penalty = 0
        self.demand = 0
    def add_inventory(self,blood):
        self.inventory.append(blood)
        
    def get_num_blood(self):
        return len(self.inventory)
        
        
    def customer_demand(self):
        noise = np.random.uniform(0.98,1)
        if self.dist == "poisson":
            self.demand = max(0,np.ceil(random.poisson(noise*self.distparams[0])))
        elif self.dist == "normal":
            self.demand = max(0,np.ceil(random.normal(noise*self.distparams[0],self.distparams[1])))
        elif self.dist == "gamma":
            self.demand = max(0,np.ceil(random.gamma(noise*self.distparams[0]*sqrt(noise*self.distparams[0])/self.distparams[1],self.distparams[1]/sqrt(noise*self.distparams[0]))))
       

    def allocate(self):
            
        self.inventory = sorted(self.inventory,key = attrgetter('shelf_life'))# since its age-based we serve the customer with the oldest products to mitigate wastages(FIFO)
        
        if self.demand <= len(self.inventory): # no shortage occured
            for i in range(int(self.demand)):
                self.inventory.pop(0) 
            fulfil_rate = self.demand
        else: # shortage occured
            self.shortage_penalty = self.shortage_cost*(self.demand - len(self.inventory))
            fulfil_rate = len(self.inventory)
            self.inventory=[]
            print("shortage penalty allert of {} dollars".format(self.shortage_penalty))
        
        return self.shortage_penalty ,fulfil_rate
            
     
        
    def update_inventory(self):
        index = []
        self.outdated = 0
        
        for i in range(len(self.inventory)): 
            self.inventory[i].update_shelf_life() # updating the self_lives at the end of each period
            if self.inventory[i].get_shelf_life() < 1:
                self.outdated += 1
                index.append(i)
        #DISPOSAL
        j = len(index)
        while j >0:
            self.inventory.pop(index[0])
            index.pop(0)
            for i in range(len(index)):
                index[i] = index[i] -1
            j = j-1
        self.wastage_penalty = self.wastage_cost * self.outdated     
        print("{} blood bags outdated at the hospital".format(self.outdated))
        print("wastage penalty allert of {} dollars".format(self.wastage_penalty))
        return self.wastage_penalty 

    #def rotate_inventory(self):
    #  if len(self.inventory) > 0:
   
'''IBTO'''
        
class Distribution_Center:
    
    def __init__(self,initial_shelf_life,wastage_cost, dist , distparams):
        self.dist = dist
        self.distparams = distparams
        self.inventory = []
        self.wastage_cost = wastage_cost
        self.initial_shelf_life = initial_shelf_life
        self.wastage_penalty = 0
        
    def add_inventory(self):
        noise = np.random.uniform(0.98,1)
        others = max(0,random.normal(noise*18.14,2.47))
        if  self.dist == "lognormal":
            self.inputs = random.lognormal(self.distparams[0],self.distparams[1]) - others
        elif   self.dist == "normal":
            self.inputs = random.normal(self.distparams[0],self.distparams[1]) - others
        elif self.dist == "weibull":
            self.inputs = self.distparams[1]*random.weibull(self.distparams[0]) - others
        elif self.dist == "gumbel":
            self.inputs = random.gumbel(self.distparams[0],self.distparams[1]) - others
  
        print("{} blood bags arrive at the distribution center".format(self.inputs))   
        for i in range(int(np.ceil(self.inputs))):
            self.inventory.append(blood(self.initial_shelf_life))
    
    def get_num_blood(self):
        return len(self.inventory)

    
    def update_inventory(self):
        index = []
        self.outdated = 0
    
        for i in range(len(self.inventory)): 
            self.inventory[i].update_shelf_life() # updating the self_lives at the end of each period
            if self.inventory[i].get_shelf_life() < 1:
                self.outdated += 1
                index.append(i)
        #DISPOSAL
        j = len(index)
        while j >0:
            self.inventory.pop(index[0])
            index.pop(0)
            for i in range(len(index)):
                index[i] = index[i] -1
            j = j-1
        print("distribution center disposes {} blood bags".format(self.outdated))  
        self.wastage_penalty = self.wastage_cost * self.outdated    
        
        return self.wastage_penalty 
    
   
'''Experimental environment'''
    
class Environment():
    
    def __init__(self,fulfil_reward,shortage_cost , wastage_cost,excess_penalty,service_reward, initial_shelf_life, HSdist=["normal","normal"],HSdistparams=[(18.03,20.1),(20.16,14.1 )],DCdist=["normal"],DCdistparams=[(55.34,4.20)],DC_number = 1,HS_number = 2,max_episode_length=120):
        self.state=[]
        self.reward = 0
        self.HS_number = HS_number
        self.DC_number = DC_number
        self.fulfil_reward = fulfil_reward
        self.shortage_cost = shortage_cost
        self.wastage_cost = wastage_cost
        self.excess_penalty = excess_penalty
        self.service_reward=service_reward
        self.initial_shelf_life = initial_shelf_life
        self.DC=[]
        self.HS=[]
        self.HSdist = HSdist
        self.HSdistparams = HSdistparams
        self.DCdist = DCdist
        self.DCdistparams = DCdistparams
        self.max_episode_length = max_episode_length
        self.done = False
        self.constraint_violation = 0
        
        for i in range(self.DC_number):
            self.DC.append(Distribution_Center(self.initial_shelf_life, self.wastage_cost,self.DCdist[i] , self.DCdistparams[i] ))
        
        for i in range(self.HS_number):
            self.HS.append(hospital(self.shortage_cost,self.wastage_cost,self.HSdist[i] , self.HSdistparams[i] ))            
        
    def get_state(self,action):
        
        state = np.zeros(shape=(self.DC_number+self.HS_number,self.initial_shelf_life))
        for j in range(1,self.initial_shelf_life+1):
          l=0
          for k in range(len(self.DC[0].inventory)):
            if self.DC[0].inventory[k].get_shelf_life()==j:
              l+=1
            state[0][j-1]=l
        for j in range(1,self.initial_shelf_life+1):
          l=0
          for k in range(len(self.HS[0].inventory)):
            if self.HS[0].inventory[k].get_shelf_life()==j:
              l+=1
            state[1][j-1]=l
        for j in range(1,self.initial_shelf_life+1):
          l=0
          for k in range(len(self.HS[1].inventory)):
            if self.HS[1].inventory[k].get_shelf_life()==j:
              l+=1
            state[2][j-1]=l
        action = np.array(action).reshape(-1)
        state = state.reshape(-1)
        inv = np.array([self.DC[0].get_num_blood(),self.HS[0].get_num_blood(),self.HS[1].get_num_blood()])
        self.state = np.concatenate((state,action,inv)).reshape([1,-1])
        return self.state
    
    def reset(self):

      self.episode_length = 0
      self.reward = 0
      self.constraint_violation = 0
      for i in range(len(self.DC)):
          self.DC[i].inventory = []
          self.DC[i].add_inventory()
          
      for i in range(len(self.HS)):
          self.HS[i].inventory = []
          
      return self.get_state(np.array([0,0]))
     
    
    # SOMETIMES THE ACTION IS INFEASIBLE THEN WE NEED TO SOLVE A QP TO FIND ITS FEASIBLE PROJECTION
    # we write it for two hospitals for now

    def project(self,action):
            if(np.sum(action)) <= 1:
              self.constraint_violation =0
              return action 
            else:
              self.constraint_violation =1
              print("we want to project the action {}".format(action))
              p = matrix(np.diag([2,2]),tc="d")
              q = matrix(np.array([2*action[0],2*action[1]]),tc="d")
              g = matrix(np.array([[1,1],[-1,0],[0,-1]]),tc="d")
              h = matrix(np.array([1,0,0]),tc="d")
              solvers.options['show_progress']=False
              solvers.options['abstol']=1e-5
              solvers.options['reltol']=1e-5
              solvers.options['feastol']=1e-5
              
              action0= solvers.qp(p,q,g,h)             
              print("the projected action is {}".format(np.floor(list(action0['x']))))
              return np.floor(list(action0['x']))
        
            
            
                       
    def step(self,action,iters):
        
    

        if iters < self.max_episode_length:
            r_c,r1,r2,r3,r4,r5 = 0,0,0,0,0,0
            #constraintrl_reward(we can handle it through the objective funtion of the actor network too)
           
                    
            #performing the action 
            # we write it for two hospitals for now
            if(np.sum(action)) > 1:
              excess = (np.sum(action) -1)* self.DC[0].get_num_blood()
              r_c = -self.excess_penalty* excess
            p_action  = self.project(action)
            self.input0 =  int(p_action[0]*self.DC[0].get_num_blood())
            self.input1 = int(p_action[1]*self.DC[0].get_num_blood())
            self.DC[0].inventory = sorted(self.DC[0].inventory,key = attrgetter('shelf_life'))
            if p_action[0] <= p_action[1]: # whichever hosipital that has greater action seems to have greater demand so we first the oldet blood bags to it
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())): 
                     self.HS[1].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())): 
                     self.HS[0].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                           
            else: 
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())): 
                     self.HS[0].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())): 
                     self.HS[1].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)

            self.HS[0].customer_demand()
            self.HS[1].customer_demand()

            for i in range(2):
              print("{} demand arrived for hospital{}".format(self.HS[i].demand,i))

            
           
                                    
            # shortages
            r1,f_rate1 = self.HS[0].allocate()
            r2,f_rate2 = self.HS[1].allocate()
            #fulfil_income
            fulfil_income = (f_rate1+f_rate2)*self.fulfil_reward
            # shortages
            r1 = -r1 
            r2 = -r2
            #wastages
            r3 = -self.HS[0].update_inventory()
            r4 = -self.HS[1].update_inventory()
            r5 = -self.DC[0].update_inventory()
            #observing the state
            self.state =  self.get_state(action)
            #construct the reward
            self.reward = r_c+r1+r2+r3+r4+r5+fulfil_income#+self.service_reward*min(f_rate1/self.HS[0].demand,f_rate2/self.HS[1].demand)
            
        else:
          r_c = 0
          if(np.sum(action)) > 1:
              excess = (np.sum(action) -1)* self.DC[0].get_num_blood()
              r_c = -self.excess_penalty* excess
         
                    
            #performing the action 
            # we write it for two hospitals for now
          p_action  = self.project(action)
          self.DC[0].inventory = sorted(self.DC[0].inventory,key = attrgetter('shelf_life'))
          if p_action[0] <= p_action[1]: # whichever hosipital that has greater action seems to have greater demand so we first the oldet blood bags to it
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())): 
                     self.HS[1].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())): 
                     self.HS[0].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                           
          else: 
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())): 
                     self.HS[0].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[0]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())): 
                     self.HS[1].add_inventory(self.DC[0].inventory[i])
                 for i in range(int(p_action[1]*self.DC[0].get_num_blood())):
                     self.DC[0].inventory.pop(0)

          self.HS[0].customer_demand()
          self.HS[1].customer_demand()
          for i in range(2):
              print("{} demand arrived for hospital{}".format(self.HS[i].demand,i))

                                
          
          r1,f_rate1 = self.HS[0].allocate()
          r2,f_rate2 = self.HS[1].allocate()
          #fulfil_income
          fulfil_income = (f_rate1+f_rate2)*self.fulfil_reward
          # shortages
          r1 = -r1 
          r2 = -r2
          #wastages
          r3 = -self.HS[0].update_inventory()
          r4 = -self.HS[1].update_inventory()
          r5 = -self.DC[0].update_inventory()
          #observing the state
          self.state =  self.get_state(action)
          #construct the reward
          self.reward = r_c+r1+r2+r3+r4+r5+fulfil_income#+self.service_reward*min(f_rate1/self.HS[0].demand,f_rate2/self.HS[1].demand)
          self.state =  self.get_state(action)
          #self.reward += -(self.HS[1].get_num_blood()+self.HS[0].get_num_blood())*self.wastage_cost
          

        self.DC[0].add_inventory()  
        return self.state , self.reward , self.HS[0].demand , self.HS[1].demand , f_rate1,f_rate2 , self.constraint_violation , self.DC[0].inputs,self.input0,self.input1,self.DC[0].outdated,self.HS[0].outdated,self.HS[1].outdated


'''FEATURIZE the approximator input(predefined state)'''

class FeatureTransformer:
    
  

  def transform(self,observations):
                                
    return observations
      
'''Hidden layer'''

# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2), dtype=np.float32)
    else:
      W = tf.random.normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)
  



'''FUNCTION APPROXIMATOR'''

# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, ft, hidden_layer_sizes=[]):
    self.ft = ft

    ##### hidden layers #####
    M1 = D
    self.hidden_layers = []
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.hidden_layers.append(layer)
      M1 = M2
      
    ''' we have two output nodes on for mean and another for variance'''
    # final layer mean
    self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)

    # final layer variance
    self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

    # get final hidden layer
    Z = self.X
    for layer in self.hidden_layers:
      Z = layer.forward(Z)

    # calculate output and cost
    mean = self.mean_layer.forward(Z)
    stdv = self.stdv_layer.forward(Z) + 1e-5 # smoothing

    # make them 1-D
    mean = tf.reshape(mean, [-1])
    stdv = tf.reshape(stdv, [-1]) 

    norm = tf.distributions.Normal(mean, stdv)
    self.predict_op = tf.clip_by_value(norm.sample(), 0, 1)#our acion must be between -1 and 1

    log_probs = norm.log_prob(self.actions)
    cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
    self.train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def sample_action(self, X):
    p = self.predict(X)[0]
    return p


# approximates V(s)
class ValueModel:
  def __init__(self, D, ft, hidden_layer_sizes=[]):
    self.ft = ft
    self.costs = []

    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1]) # the output
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    self.cost = cost
    self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
    self.costs.append(cost)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})



        
''' play one td_learning'''

def play_one_td(env, pmodel1,pmodel2 ,vmodel, gamma,episode_number):

  #print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
  print ("ep{}__".format(episode_number))
  observation = env.reset()
  done = False
  total_cons_violate=0
  totalreward = 0
  total_demand1=0
  total_demand2=0
  total_demand = 0
  total_fulfil1=0
  total_fulfil2=0
  total_fulfil=0
  total_input=0#system input
  total_input1=0# sent blood to hospital one
  total_input2=0# sent blood to hospital two
  total_waste1 = 0 #wasted at DC
  total_waste2=0 # wasted HS1
  total_waste3 =0 #wasted at HS2
  total_waste=0 #wasted at the system
  iters = 0

  for i in range(env.max_episode_length) :
    iters +=1
    print ("ep{}__".format(episode_number))
    #print("current state is = {}".format(observation))
    action1 = pmodel1.sample_action(observation)
    action2 = pmodel2.sample_action(observation)
    action = [action1,action2]
    print('the suggested action by the agents is {}'.format(action))
    prev_observation = observation
    observation, reward ,demand1,demand2,fulfil1,fulfil2,cons_violate,Input,Input1,Input2,waste1,waste2,waste3 = env.step(action,iters)
    print(reward)
    #print ("next state is = {}".format(observation))
    #print("reward is = {}".format(reward))
    
    '''
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print ("ep{}__".format(episode_number),"iter:{}__".format(iters),"inventory_constraint: {}".format(initial_inventory) , "action:{}".format(action), "demand:(H1 : {}, H2 : {})".format(env.HS[0].demand,env.HS[1].demand) , "wastage:(H1: {}, H2: {}, DC: {})".format(env.HS[0].outdated , env.HS[1].outdated ,env.DC[0].outdated), "shortage:(H1 : {}, H2 : {})".format(env.HS[0].demand-action[0],env.HS[1].demand-action[1]))
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    '''
    total_cons_violate += cons_violate
    totalreward += reward
    total_demand1 += demand1
    total_demand2 += demand2
    total_fulfil1 += fulfil1
    total_fulfil2 += fulfil2
    total_demand += demand1 + demand2
    total_fulfil += fulfil1 + fulfil2
    total_input += Input
    total_input1 += Input1
    total_input2 += Input2
    total_waste1 += waste1
    total_waste2 += waste2 
    total_waste3 += waste3 
    total_waste += waste1+waste2+waste3
    
    if iters < env.max_episode_length:
      # update the models
      V_next = vmodel.predict(observation)
     
      #print('V_next = {}'.format(V_next))
      G = reward + gamma*V_next
      #print('G = {}'.format(G))
      baseline = vmodel.predict(prev_observation)
      #print("baseline = {}".format(baseline))
      advantage = G - baseline
      #print('advantage = {}'.format(advantage))
      
      pmodel1.partial_fit(prev_observation,action1, advantage)
      pmodel2.partial_fit(prev_observation,action2, advantage )
      vmodel.partial_fit(prev_observation, G)
    
    elif iters == env.max_episode_length:
       G = reward  
       #print('G = {}'.format(G))
       advantage = G - vmodel.predict(prev_observation)
       #print('advantage = {}'.format(advantage))
       pmodel1.partial_fit(prev_observation, action1, advantage )
       pmodel2.partial_fit(prev_observation, action2, advantage )
       vmodel.partial_fit(prev_observation, G)
      
       
  service_level1 = total_fulfil1/total_demand1
  service_level2 = total_fulfil2/total_demand2
  service_level_total = total_fulfil/total_demand
  waste_rate1 = total_waste1/total_input
  if total_input1 != 0 :
      waste_rate2 = total_waste2/total_input1
  else:# no blood was sent to it to be wasted
      waste_rate2=0
  if total_input2 != 0 :
      waste_rate3 = total_waste3/total_input2
  else:# no blood was sent to it to be wasted
      waste_rate3 = 0
  waste_rate_total = total_waste/total_input
  return totalreward, iters,service_level1,service_level2,service_level_total ,total_cons_violate,waste_rate1,waste_rate2,waste_rate3,waste_rate_total,total_demand1,total_demand2

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Average of the Earned reward over the last 100 episodes")
  plt.ylabel("Running Average")
  plt.xlabel("Episodes")
  plt.show()     
def plot_running_avg_of_service(s1,s2,st):
  N = len(s1)
  running_avg1 = np.empty(N)
  running_avg2 = np.empty(N)
  running_avg_total = np.empty(N) 
  for t in range(N):
    running_avg1[t] = s1[max(0, t-100):(t+1)].mean()
    running_avg2[t] = s2[max(0, t-100):(t+1)].mean()
    running_avg_total[t] = st[max(0, t-100):(t+1)].mean()   
  plt.plot(running_avg1,label="HS1_SL")
  plt.plot(running_avg2,label="HS2_SL")
  plt.plot(running_avg_total,label="Total_SL")
  plt.ylabel("Average of the Last 100 Episodes' Service levels")
  plt.xlabel("Episodes")
  plt.title("Service Level")
  plt.legend()
  plt.show()  
def plot_running_avg_violate(totalviolations):
  N = len(totalviolations)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalviolations[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Average of the Last 100 Episodes' Constraint Violations")
  plt.ylabel("Average number of the Constraint violations")
  plt.xlabel("Episodes")
  plt.show()     
def plot_running_avg_of_waste(s1,s2,s3,st):
  N = len(s1)
  running_avg1 = np.empty(N)
  running_avg2 = np.empty(N)
  running_avg3 = np.empty(N)
  running_avg_total = np.empty(N) 
  for t in range(N):
    running_avg1[t] = s1[max(0, t-100):(t+1)].mean()
    running_avg2[t] = s2[max(0, t-100):(t+1)].mean()
    running_avg3[t] = s3[max(0, t-100):(t+1)].mean()
    running_avg_total[t] = st[max(0, t-100):(t+1)].mean()   
  plt.plot(running_avg1,label="DC_WR")
  plt.plot(running_avg2,label="HS1_WR")
  plt.plot(running_avg3,label="HS2_WR")
  plt.plot(running_avg_total,label="Total_WR")
  plt.ylabel("Average of the Last 100 Episodes' Wastage rates")
  plt.xlabel("Episodes")
  plt.title("wastage rate")
  plt.legend()
  plt.show()
def plot_running_avg_of_demand(s1,s2):
  N = len(s1)
  running_avg1 = np.empty(N)
  running_avg2 = np.empty(N) 
  for t in range(N):
    running_avg1[t] = s1[max(0, t-100):(t+1)].mean()
    running_avg2[t] = s2[max(0, t-100):(t+1)].mean() 
  plt.plot(running_avg1,label="HS1_AVERAGE OF Total demand")
  plt.plot(running_avg2,label="HS2_AVERAGE OF Total demand")
  plt.ylabel("Average of the Last 100 Episodes' Total Demands")
  plt.xlabel("Episodes")
  plt.title("Demand")
  plt.legend()
  plt.show()

'''defining the envs'''
initial_shelf_life = 30
env = Environment(fulfil_reward =0.2, shortage_cost=2, wastage_cost=1,excess_penalty=1,service_reward=0,initial_shelf_life=initial_shelf_life)
ft = FeatureTransformer()
D = 2+3+(3*initial_shelf_life)
pmodel1 = PolicyModel(D,ft,[40,40,40])
pmodel2 = PolicyModel(D,ft,[40,40,40])
vmodel = ValueModel(D,ft,[40,40,40])
init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)
pmodel1.set_session(session)
pmodel2.set_session(session)
vmodel.set_session(session)
gamma = 1


N = 500
totalrewards = np.empty(N)
service_level1s = np.empty(N)
service_level2s = np.empty(N)
service_level_totals = np.empty(N)
constraint_violations = np.empty(N)
waste_rates1 = np.empty(N)
waste_rates2 = np.empty(N)
waste_rates3 = np.empty(N)
waste_rate_totals = np.empty(N)
demands1=np.empty(N)
demands2 = np.empty(N)
for n in range(N):
  totalreward, num_steps,service_level1,service_level2,service_level_total,total_cons_violate,waste_rate1,waste_rate2,waste_rate3,waste_rate_total,total_demand1,total_demand2  = play_one_td(env, pmodel1,pmodel2,vmodel, gamma,n)
  totalrewards[n] = totalreward
  service_level1s[n] = service_level1
  service_level2s[n] = service_level2
  service_level_totals[n] = service_level_total
  constraint_violations[n] = total_cons_violate
  waste_rates1[n] = waste_rate1
  waste_rates2[n] = waste_rate2
  waste_rates3[n] =  waste_rate3
  waste_rate_totals[n] =  waste_rate_total
  demands1[n]=total_demand1
  demands2[n]=total_demand2
 
print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
#
plt.plot(totalrewards[:])
plt.xlabel("Episodes")
plt.ylabel("Earned Rewards per episode")
plt.title("Rewards")
plt.show()
plot_running_avg(totalrewards[:])
#
plot_running_avg_of_service(service_level1s[:],service_level2s[:],service_level_totals[:])
#
plt.plot(constraint_violations[:])
plt.xlabel("Episodes")
plt.ylabel("Total Number of Constraint Violations per episode")
plt.title("Total Number of Inventory Constraint Violations")
plt.show()
plot_running_avg_violate(constraint_violations[:])
#
plot_running_avg_of_waste( waste_rates1[:],waste_rates2[:],waste_rates3[:] ,waste_rate_totals[:])
#
plt.plot(demands1[:]) 
plt.ylabel("demand of Hospital one")
plt.xlabel("Episodes")
plt.title("HS1 Demands")
plt.show()

plt.plot(demands2[:])  
plt.ylabel("demand of Hospital one")
plt.xlabel("Episodes")
plt.title("HS2 Demands")
plt.show()
plot_running_avg_of_demand(demands1[:],demands2[:])
