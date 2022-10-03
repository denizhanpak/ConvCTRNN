import ctrnn
import evolutionary_algorithm
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mndata = MNIST("mnist_data")
images, labels = mndata.load_training()

to_use = []

for i, image in enumerate(images):
    label = labels[i]
    if label in [1,2]:
        to_use.append((label,image))


def task(genome, count = 2):
    score = 0
    agent = ctrnn.ctrnn_from_genome(genome)
    for label, image in to_use[:count]:
        pred = simulate(3, agent, image)
        if pred == label:
            score += 1
    return score/count

def simulate(seconds, agent, image, dt=0.1):
    image = (np.array(image).reshape(28,28))
    agent_pos = (14,14)
    velocity_constant = 0.2
    image = np.pad(image,(1,1), constant_values =(0))
    for step in range(seconds):
        if agent_pos == (1,1):
            return 1
            
        if agent_pos == (28,28):
            return 2
        
        agent_x = agent_pos[0]
        agent_y = agent_pos[1]
        agent.observe(image[agent_x-1:agent_x+1,agent_y-1:agent_y+1])
        
        for i in range(int(1/dt)):
            agent.step(dt)

        output = agent.Outputs[:4]
        if (np.isnan(output)).any():
            return 0
        lateral = int((output[0] - output[1])/velocity_constant)
        vertical = int((output[2] - output[3])/velocity_constant)

        agent_pos = (agent_pos[0] + lateral, agent_pos[1] + vertical)
        agent_pos = tuple(np.clip(agent_pos,1,28))
        
        #visual = np.copy(image)
        #visual[agent_x-1:agent_x+1,agent_y-1:agent_y+1] = 150
        #ax = plt.gca()V
        #rect = Rectangle((3,3)
        #plt.imshow(visual)
        #plt.pause(0.01)
    
    return 0


agent = ctrnn.CTRNN(6)
agent.randomizeParameters()
algo = evolutionary_algorithm.Microbial(task, 
        genesize=agent.gene_size(), popsize=500,generations=1000)
algo.run()
avgfit, bestfit, bestind = algo.fitStats()
print(bestind, bestfit)
algo.showFitness()
algo.saveFitness("results")
