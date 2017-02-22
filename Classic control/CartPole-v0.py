import gym
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import BiasUnit
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
import time
import numpy as np
import copy
import random
import operator
from gym import wrappers
import math

poolSize = 14;
pool = [];
elitismSize = 4;
crossRate = 0.9;
mutRate = 0.25;

def newNet(inLen, outLen):
    n = FeedForwardNetwork()

    bias = BiasUnit(name='bias')
    n.addModule(bias)

    inLayer = LinearLayer(inLen, name='in')
    hiddenLayer = LSTMLayer(20, name='hidden')
    outLayer = LinearLayer(outLen, name='out')

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    bias_to_hidden = FullConnection(bias, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    n.addConnection(in_to_hidden)
    n.addConnection(bias_to_hidden)
    n.addConnection(hidden_to_out)

    n.sortModules()
    return n;

def newNet2(inLen, outLen):
    return buildNetwork(inLen, 20, 20, outLen, hiddenclass=TanhLayer);

def crossOver(parent1, parent2):
    global crossRate;
    
    if random.random() <= crossRate:
        child1 = copy.deepcopy(parent1);
        child2 = copy.deepcopy(parent2);
        
        for thing in parent1.modules:
            randy = thing.name;
            lll = len(parent1.connections[parent1[randy]]);
            
            for _x in xrange(lll):
                found = -1;
                inmod_name = parent1.connections[parent1[randy]][_x].inmod.name
                outmod_name = parent1.connections[parent1[randy]][_x].outmod.name
                
                for _y in range(lll):
                    if inmod_name == parent2.connections[parent2[randy]][_y].inmod.name and outmod_name ==  parent2.connections[parent2[randy]][_y].outmod.name:
                        found = _y;
                        break;
                        
                temp1 = parent1.connections[parent1[randy]][_x].params[:];
                temp2 = parent2.connections[parent2[randy]][found].params[:];
                
                cutLocation = random.randint(0, len(temp1));
                
                param1 = np.concatenate((temp1[0:cutLocation], temp2[cutLocation:]));
                param2 = np.concatenate((temp2[0:cutLocation], temp1[cutLocation:]));
                
                child1.connections[child1[randy]][_x]._setParameters(param1, child1.connections[child1[randy]][_x].owner);
                child2.connections[child2[randy]][found]._setParameters(param2, child2.connections[child2[randy]][found].owner);
                
        return child1, child2;
    return parent1, parent2;

def mutate(parent):
    global mutRate;
    if random.random() <= mutRate:
        for mod in parent.modules:
            for conn in parent.connections[mod]:
                x = conn.params;
                for _ in xrange(len(x)):
                    x[_] += x[_] * (random.random() - 0.5) * 3 + (random.random() - 0.5);

def selection(pool):
    total = 0.0;
    
    for chromo in pool:
        total = total + chromo.fitness;
        
    sli = float(total) * random.random();
    ttt = 0.0;
    
    for xxx in xrange(len(pool)-1, -1, -1):
        ttt = ttt + pool[xxx].fitness;
        if ttt >= sli:
            #remove this one and return it;
            return pool.pop(xxx);
            
    return pool.pop();

def initList():
    global pool;
    for _ in xrange(poolSize):
        genome = newNet2(len(env.observation_space.high), env.action_space.n);
        pool.append(genome);


env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
maxxy = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
episode_num = 0;
first = True;
cum_reward = [];
avg_reward = float(0);
gen_num = 0;

if first:
    initList();
    first = False;

while True:
    newpool = [];
    print "Testing Generation #" + str(gen_num);
    for _ in xrange(poolSize):
        genome = pool[_];
        ep_reward = 0.0;
        observation = env.reset()
        
        for __ in xrange(maxxy):
            #env.render()
            result = genome.activate(observation)
            action = np.argmax(result)
            observation, reward, done, info = env.step(action)
            ep_reward += reward;
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                episode_num += 1;
                break
                
        cum_reward.append(ep_reward);
        genome.fitness = ep_reward;
    
    if len(cum_reward) >= 100:
        cum_reward = cum_reward[-100:];
        avg_reward = np.mean(cum_reward[:]);        
        print "Average Award:", avg_reward;
        
    if avg_reward >= 195.0:
        #we did it;
        print("Success achieved after {} episodes".format(episode_num)); 
        break;
            
    #all gens done create new gen
    sortedpop = sorted(pool, key=lambda x: x.fitness, reverse=True);
    elite = sortedpop[0:elitismSize];
    
    for m in xrange(len(pool)-1,-1,-2):
        p1 = selection(pool);
        p2 = selection(pool);
        ch1, ch2 = crossOver(p1, p2);
        #mutate(ch1);
        #mutate(ch2);
        newpool.append(ch1);
        newpool.append(ch2);
        
    newpool = random.sample(newpool, poolSize - elitismSize);
    pool = [];
    pool = reduce(operator.add, [elite, newpool]);
    gen_num += 1;

env.close();
gym.upload('/tmp/cartpole-experiment-1', api_key='your_api_key_here')
