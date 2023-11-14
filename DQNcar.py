from env.autocar import *
from DQN.algo import DQN
from DQN.model import Model,DeepQNetwork
from DQN.my_agent import Agent
from DQN.replay_memory import ReplayMemory
LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 50000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 500  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.95  # reward 的衰减因子，一般取 0.9 到 0.999 不等


def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()[0]
    step = 0
    while True:
        #先采样数据预热rpm
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done= env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


if __name__ == '__main__':

    run = True
    clock = pygame.time.Clock()
    images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
            (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
    env = ComputerCar(1, 1, PATH)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    rpm = ReplayMemory(MEMORY_SIZE)
    model = DeepQNetwork(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim=act_dim,
        e_greed=0.05,  # 有一定概率随机选取动作，探索
        e_greed_decrement=5e-4)  # 随着训练逐步收敛，探索的程度慢慢降低

    scores = []
    n_games = 10000
    run = True
    win_num = 0
    for i in range(n_games):
        score = 0
        idx = 0
        done = False
        obs = env.reset()
        if win_num >10:
            print("game will finish here because the paras are best!")
            break
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
            model.train()
            action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
            next_obs, reward, done = env.step(action)
            rpm.append((obs, action, reward, next_obs, done))

            # train model
            if (len(rpm) > MEMORY_WARMUP_SIZE) and (idx%LEARN_FREQ == 0):
                print("learn")
                # s,a,r,s',done
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                 batch_done) = rpm.sample(BATCH_SIZE)
                train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                         batch_next_obs, batch_done)
            score += reward
            obs = next_obs
            idx += 1
            # print(observation)
        print('episodes: ' + str(i) + '------score: ' + str(score)+'------win num: '+str(win_num))
        if score > 100000:
            win_num += 1
        if i > 100:
            pass
            # TODO: save the trained network weights
            ######################################
            # torch.save(agent.Q_eval.state_dict(), 'weight_eval.pt')
            # torch.save(agent.Q_next.state_dict(), 'weight_next.pt')
            # agent.memory.save_buffer('buffer')
            ######################################
        if run != True:
            break
        # scores.append(score)

        # avg_score = np.mean(scores[-100:])

    save_path = './dqn_car_model.pth'
    algorithm.save(save_path)
    pygame.quit()

