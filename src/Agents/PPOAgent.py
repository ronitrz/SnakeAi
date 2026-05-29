from BaseAgent import BaseAgent

class PPO(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        
    
    def act(self):
        ...
        
    def observe(self, reward, done):
        ...
        
    