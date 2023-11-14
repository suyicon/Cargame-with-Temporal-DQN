import torch

from env.autocar import *
from DQN.algo import DQN
from DQN.my_agent import Agent
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.95  # reward 的衰减因子，一般取 0.9 到 0.999 不等
if __name__ == '__main__':

    run = True
    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
            (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
    env = ComputerCar(1, 1, PATH)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    model = torch.load("dqn_car_model.pth")
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim=act_dim,
        e_greed=0,
        e_greed_decrement=0)

    scores = []
    n_games = 100
    run = True
    win_num = 0
    for i in range(n_games):
        score = 0
        idx = 0
        done = False
        obs = env.reset()

        '''
        observation
        1. beta: sideslip angle (the angle between heading angle of the car and the center line)
        2. deviation: deviation between car and center line
        3. direction: check the car's direction from the center line(1 for left and -1 right) 
        '''

        while not done:
            clock.tick(60)
            draw(WIN, images, env)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            if run != True:
                break
            model.eval()
            action = agent.predict(obs)
            next_obs, reward, done = env.step(action)
            score += reward
            obs = next_obs
            idx += 1
            # print(observation)
        print('episodes: ' + str(i) + '------score: ' + str(score)+'------win num: '+str(win_num))
        if score > 100000:
            win_num += 1
        if i > 100:
            pass
        if run != True:
            break
        # scores.append(score)

        # avg_score = np.mean(scores[-100:])