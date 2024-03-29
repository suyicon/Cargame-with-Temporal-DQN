import numpy as np
import pygame
import math
from env.utils import scale_image, blit_rotate_center

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track1.png"),0.3)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track_border1.png"),0.3)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (453, 410)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.4)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.3)
CENTER_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.05)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

FPS = 30
PATH = [(501, 370), (501, 314), (498, 221), (462, 142), (403, 93), (325, 73), (247, 96), (184, 152), (149, 237), (148, 353),
        (150, 459), (148, 565),
        (148, 672), (180, 771), (238, 828), (324, 853), (409, 832), (470, 776), (501, 678), (503, 568), (500, 516),
        (499, 467)]


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.obs_dim = 3
        self.act_dim = 2
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
            # print('left')

        elif right:
            self.angle -= self.rotation_vel
            # print('right')

        if self.angle > 180:
            self.angle -= 360
        elif self.angle < -180:
            self.angle += 360

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()
        # print('forward')

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()
        # print('backward')

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class ComputerCar(AbstractCar):
    IMG = GREEN_CAR
    START_POS = (488, 370)

    def __init__(self, max_vel, rotation_vel, path=[]):
        super().__init__(max_vel, rotation_vel)
        self.path = path
        self.current_point = 0
        self.car_rect = (0, 0, 0, 0)
        self.is_finished = False
        self.is_collide = False
        self.cumulated_rewards = 0
        self.__calculate_x_y()

        self.idx = 0
        self.dist_ls = [[0, 0], [0, 0]]

        self.direc = 0  # for pid control
        self.target = []
        for i in self.path:
            rect = pygame.Rect(0, 0, 5, 20)
            rect.center = i
            self.target.append(rect)

    def __calculate_x_y(self):
        self.cx = self.x + self.img.get_width() / 2
        self.cy = self.y + self.img.get_height() / 2

    def __calculate_dist(self):
        target_x, target_y = self.path[self.current_point]
        x_diff = target_x - self.cx
        y_diff = target_y - self.cy
        if self.idx >= 5:
            self.dist_ls[0][0] = self.dist_ls[1][0]
            self.dist_ls[0][1] = self.dist_ls[1][1]
            self.dist_ls[1][0] = x_diff
            self.dist_ls[1][1] = y_diff
            self.idx = 0
        self.idx += 1

        ########### calculate the deviation of the car from center line ###########
        A_x, A_y = self.path[self.current_point - 1]
        B_x, B_y = self.path[self.current_point]
        C_x = self.cx
        C_y = self.cy

        # verctor c
        dc_x = B_x - A_x
        dc_y = B_y - A_y

        # vector b
        db_x = C_x - A_x
        db_y = C_y - A_y

        cross_prdct = db_x * dc_y - db_y * dc_x
        if cross_prdct > 0:
            direction = 1
        elif cross_prdct == 0:
            direction = 0
        else:
            direction = -1
        self.direc = direction

        # angle of vector b and c
        c = math.sqrt(dc_x ** 2 + dc_y ** 2)
        b = math.sqrt(db_x ** 2 + db_y ** 2)
        theta = math.acos((dc_x * db_x + dc_y * db_y) / (b * c))
        # deviation
        dev = abs(b * math.sin(theta))
        ##############################################################################
        ############### calculate sideslip angle beta ##################
        # angle between the Frenet and the ground
        # let vector C to be the axis S of the Frenet
        if dc_y != 0:
            phi = math.atan(dc_x / dc_y) * 180 / math.pi
            if dc_y < 0:
                phi = phi
            elif dc_x < 0:
                phi = 180 - abs(phi)
            else:
                phi = -180 + abs(phi)
        else:
            if dc_x < 0:
                phi = 90
            else:
                phi = -90

        # beta = self.angle - phi
        # print(phi, self.angle, dc_x,dc_y)
        beta = self.angle - phi
        if (beta > 180) or (beta < -180):
            if phi >= 0:
                beta = 360 - abs(beta)
            else:
                beta = abs(beta) - 360
        #################################################################

        return beta, dev, direction

    # define the environment rewards
    def __get_rewards(self):
        # is collided
        car_mask1 = pygame.mask.from_surface(self.img)
        offset = (int(self.x - 0), int(self.y - 0))
        poi = TRACK_BORDER_MASK.overlap(car_mask1, offset)
        if poi != None:
            is_collided = 1
            self.is_collide = True
        else:
            is_collided = 0
            self.is_collide = False

        beta, dev, direction = self.__calculate_dist()

        dist_0 = math.sqrt(self.dist_ls[0][0] ** 2 + self.dist_ls[0][1] ** 2)
        dist_1 = math.sqrt(self.dist_ls[1][0] ** 2 + self.dist_ls[1][1] ** 2)

        ddist = dist_1 - dist_0  # check that is the car moving

        # TODO: configurate the reward
        if dev < 2:
            rewards = dev * 0.3 - 0.05 * abs(beta) - 1 / (ddist * 10 + 200)
        else:
            rewards = -dev * 0.3 - 0.05 * abs(beta) - 1 / (ddist * 10 + 200)

        if is_collided:
            rewards = rewards - 2000
            self.reset()

        if self.to_target:
            if self.current_point > 3:
                print("to target", self.current_point)
                rewards =1000
            elif self.current_point > 6:
                print("to target", self.current_point)
                rewards = 1200
            elif self.current_point > 9:
                print("to target", self.current_point)
                rewards =1400
            elif self.current_point > 12:
                print("to target", self.current_point)
                rewards =1600
            elif self.current_point > 15:
                print("to target", self.current_point)
                rewards =1800
            else:
                print("to target",self.current_point)
                rewards = 800

        if self.is_finished:
            rewards = 100000
        return rewards, beta, dev, direction

    def step(self, keys):
        moved = False

        if keys == 0:
            self.rotate(left=True)
            moved = True
            self.move_forward()
        if keys == 1:
            self.rotate(right=True)
            moved = True
            self.move_forward()
        # if keys == 2:
        #     moved = True
        #     self.move_forward()
        # if keys == 3:
        #     moved = True
        #     self.move_backward()

        if not moved:
            self.reduce_speed()
        self.__calculate_x_y()
        self.__handle_collision()

        reward, beta, dev, direction = self.__get_rewards()
        self.cumulated_rewards += reward

        # if self.cumulated_rewards < -1000:
            #done = True
        if self.is_finished:
            done = True
        elif self.is_collide:
            done = True
        else:
            done = False

        # print(self.__sensor(34,24,4))
        # print('reward: ' + str(self.__get_rewards()))
        return ([beta, dev, direction], reward, done)

    def __handle_collision(self):
        if self.collide(TRACK_BORDER_MASK) != None:
            self.bounce()
        player_finish_poi_collide = self.collide(
            FINISH_MASK, *FINISH_POSITION)
        if player_finish_poi_collide != None:
            if player_finish_poi_collide[1] == 0:
                self.bounce()
            else:
                self.reset()
                self.is_finished = True

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def draw_points(self, win):
        for rect in self.target:
            pygame.draw.rect(win, (230, 230, 230), rect)
        self.__calculate_x_y()
        pygame.draw.circle(win, (230, 230, 230), (self.cx, self.cy), 2)

    def draw_car_rect(self, win):
        pygame.draw.rect(win, (230, 230, 230), self.car_rect, 1)

    def draw(self, win):
        super().draw(win)
        self.draw_points(win)
        self.draw_car_rect(win)

    def bounce(self):
        self.vel = -self.vel * 0.7
        self.move()

    def update_path_point(self):
        to_target = False
        if self.current_point >= len(self.path) - 1:
            self.current_point = 0
            return
        target = self.path[self.current_point]
        #target = self.target[self.current_point]
        rect = pygame.Rect(
            self.x - 15, self.y - 10, self.img.get_width() + 30, self.img.get_height() + 20)

        self.car_rect = ((rect[0], rect[1]), (rect[2], rect[3]))
        # print(self.car_rect,self.current_point)
        if rect.collidepoint(*target):
            self.current_point += 1
            to_target = True
        self.to_target = to_target

    def move(self):
        self.update_path_point()
        super().move()

    def reset(self):
        self.is_finished = False
        self.current_point = 0
        self.cumulated_rewards = 0
        super().reset()
        return np.array([0, 0, 0])


def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    pygame.display.update()
