

import gym
from gym import spaces
import numpy as np
import pygame

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, size = 10):
        super(MazeEnv, self).__init__()

        self.size = size
        self.window_size = 512
        self.action_space = spaces.Discrete(4)  # 4 move ; up, down, left, right

        self.observation_space = spaces.Dict(
            {
                "agent" : spaces.Box(0, size - 1, shape = (2,), dtype = int),
                "Exit" : spaces.Box(0, size - 1, shape = (2,), dtype = int)
            }
        )

        self.maze = np.zeros(self.size)
        self.robot_position = np.array([0, 0])  # agent start position
        self.Exit_position = np.array([9, 9])  # Exit position
        self.wall_positions = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                                        [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
                                        [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [8, 2],
                                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],
                                        [1, 3], [2, 2], [2, 5], [2, 6], [3, 5], [4, 3], [1, 4], [4, 7],  
                                        [5, 1], [5, 6], [5, 7], [6, 1], [6, 3], [6, 5], [6, 6], [7, 8]])

        self.blackhole = np.array([5, 8])
        
        self.action_move = {
            0 : np.array([1, 0]),   # up
            1 : np.array([-1, 0]),  # down
            2 : np.array([0, 1]),   # right
            3 : np.array([0, -1])   # left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None
        self.wall_map = None
        self.blackhole_map = None


    def get_obs(self) :
        return {
            "agent" : self.robot_position,
            "Exit" : self.Exit_position
        }
    
    def get_info(self) :
        return {
            "distance" : np.linalg.norm(
                self.robot_position - self.Exit_position, ord = 1
            )
        }
    
    def reset(self, seed = None):
        super().reset(seed = seed)
        
        self.wall_positions[39] = [1, 4]
        self.wall_map = np.zeros((self.size, self.size), dtype = bool)
        for pose in self.wall_positions :
            self.wall_map[pose[0], pose[1]] = True
        
        self.robot_position = np.array([0, 0])  # return to the start point
        self.maze = np.zeros(self.size)
        self.Exit_position = np.array([9, 9])
        self.blackhole = np.array([5, 8])
        self.move_time1 = 1
        self.move_time2 = 1
        self.wall_positions[39] = [1, 4]

        observation = self.get_obs()

        # Update wall_map after modifying wall_positions
        for pose in self.wall_positions:
            self.wall_map[pose[0], pose[1]] = True
        
        if self.render_mode == "human":
            self.render_frame()

        return observation
    
    def test(self) :

        if self.move_time1 % 2 == 1:
            if np.array_equal(self.wall_positions[39], [8, 4]) :
                self.wall_positions[39][0] -= 1 # from [1, 4]
                #self.wall_positions[40][0] -= 1 # from [2, 4]
                self.move_time1 += 1    
            else :  
                self.wall_positions[39][0] += 1
                #self.wall_positions[40][0] += 1
        else :
            if np.array_equal(self.wall_positions[39], [1, 4]) :
                self.wall_positions[39][0] += 1 
                #self.wall_positions[40][0] += 1
                self.move_time1 += 1
            else :
                self.wall_positions[39][0] -= 1
                #self.wall_positions[40][0] -= 1
        
        if self.move_time2 % 2 == 1:
            if np.array_equal(self.wall_positions[47], [8, 6]) :
                self.wall_positions[47][0] -= 1 
                self.move_time2 += 1    
            else :  
                self.wall_positions[47][0] += 1
        else :
            if np.array_equal(self.wall_positions[47], [6, 6]) :
                self.wall_positions[47][0] += 1 
                self.move_time2 += 1
            else :
                self.wall_positions[47][0] -= 1

        # Update wall_map after modifying wall_positions
        self.wall_map = np.zeros((self.size, self.size), dtype=bool)
        for pose in self.wall_positions:
            self.wall_map[pose[0], pose[1]] = True

    def step(self, action):
        
        count = 0
        died = 0
 
        # update location # using 'np.clip' to make sure we don't leave map
        new_location = np.clip(
            self.robot_position + self.action_move[action], 0, self.size - 1
        )
        
        location = [new_location[1], new_location[0]]

        avoid_collision = self.avoid_collision(location, self.wall_positions)
        
        if avoid_collision == 1 :
            self.robot_position = self.robot_position
            count = 1
        else :
            self.robot_position = new_location
            count = 2

        # calculate reward
        if count == 1 : # which mean not moving (next position is wall)
            reward = - 5
        elif np.array_equal(self.robot_position, self.Exit_position) :
            reward = + 50
        elif np.array_equal(self.robot_position, self.blackhole) :
            reward = - 100
            died = 1
        else :
            reward = - 1

        observation = self.get_obs()
        
        if self.render_mode == "human":
            self.render_frame()

        info = {"died" : died}

        return observation, reward, reward > 0, False, info,

    def avoid_collision(self, a, b):
        c = 0
        for i in range(len(b)):
            if np.array_equal(a, b[i]):
                c = c + 1
        if c == 0 :
            return 0
        else :
            return 1
       
    def location(self, action) :
        new_location = np.clip(
            self.robot_position + self.action_move[action], 0, self.size - 1
        )

        location = new_location
        location[0] = new_location[1]
        location[1] = new_location[0]

        avoid_collision = self.avoid_collision(location, self.wall_positions)
        
        if avoid_collision == 1 :
            self.robot_position = self.robot_position
            count = 1
        else :
            self.robot_position = new_location
            count = 2
        
        return location, new_location
     

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()
    
    def render_frame(self) :
        if self.window is None and self.render_mode == "human" :
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human" :
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # The size of a single grid in pixels
        pix_square_size = (self.window_size / self.size)
        
        # draw walls
        for i in range(self.size) :
            for j in range(self.size) :
                if self.wall_map[i, j] :
                    pygame.draw.rect(
                        canvas,
                        (128, 128, 128),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        )
                    )

        # draw the Exit
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.Exit_position,
                (pix_square_size, pix_square_size)
            )
        )

        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.robot_position + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        # add gridlines
        for x in range(self.size + 1) :
            # x axis
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width = 3
            )
            # y axis
            pygame.draw.line(
                canvas,
                0,                                     
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width = 3 
            )

        # draw blackhole
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self.blackhole + 0.5) * pix_square_size,
            pix_square_size / 3
        )
                   
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self) :
        if self.window is not None :
            pygame.display.quit()
            pygame.quit()

