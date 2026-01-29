class DQN(nn.Module):
    def __init__(self):
        pass


"""
use policynet -> expect[action] we get action on the bases of prob ->argmax to find best one
prep dict -> to store mry 
train -> take place -> policy network
q val comp against arget_q_val [obtained from detached -> block dl/dw] as err

"""