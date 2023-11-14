import numpy as np
import torch


class Agent:
    def __init__(self,algo,act_dim,e_greed = 0.1,e_greed_decrement = 0):
        self.act_dim = act_dim
        self.global_step = 0
        self.update_target_steps = 200
        self.algo = algo
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = np.expand_dims(obs, axis=-2)
        obs = torch.FloatTensor(obs)
        pred_q = self.algo.predict(obs)
        pred_q = pred_q.squeeze(axis=0)
        act = int(pred_q.argmax())
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        if self.global_step % self.update_target_steps == 0:
            self.algo.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.FloatTensor(obs)
        obs = obs.unsqueeze(0)
        #print("obs",obs.shape)

        act = torch.IntTensor(act)
        reward = torch.FloatTensor(reward)
        next_obs = torch.FloatTensor(next_obs)
        terminal = torch.FloatTensor(terminal)
        loss = self.algo.learn(obs, act, reward, next_obs, terminal)
        return float(loss)