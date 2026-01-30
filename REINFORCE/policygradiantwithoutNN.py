import numpy as np 

theta = 0.0 
alpha = 0.01 

def policy(action):
    prob_right = 1/(1+np.exp(-theta)) #sigmoid fun
    return prob_right if action ==1 else 1- prob_right 

#print(np.random.choice([2,4,8],20,p=[0.3,0.1,0.6]))

#sample an action from policy
def sample_action():
    return np.random.choice([0,1],p=[policy(0),policy(1)])

def environment(action):
    return 1 if action==1 else 0

for episode in range(100):
    action=sample_action()
    reward=environment(action)

    #compute gradient update
    grad_log_policy=action-policy(action) #gradient of log 
    theta+=alpha*grad_log_policy*reward 

    if episode%10==0:
        print(f"Episode {episode}, theta = {theta:.4f}, prob_right = {policy(1):.4f}")