import time
import pygame
import numpy as np
from collections import deque
import math


WIDTH = 10
LENGTH = 10

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)

RED = (199, 5, 0)
GREEN = (34, 139, 34)

GAME_SIZE = 1_000

pygame.init()
screen = pygame.display.set_mode((WIDTH * 20, LENGTH * 20))


class CGOL:

    def __init__(self, w=LENGTH, h=WIDTH, size=20) -> None:
        self.w = w
        self.h = h
        self.size = size
        self.game_iteration = 0 # number of simulations ran
        # self.unique_states = []
        self.running = True
        self.num_states = 0 # number of states reached in one game


        self.cells = np.random.randint(0, 2, (WIDTH, LENGTH))
        self.log = deque(maxlen=20)
        self.reached = np.array([], dtype=int)

        screen.fill(COLOR_GRID)
        self.cells = self.update_ui(screen, size, with_progress=True)

        pygame.display.update()

    
    def get_state(self):
        return self.seed
    
    def get_avg_reached(self):
        print(f'count: {(self.reached).size}')
        print(f'average states reached: {np.average(self.reached)}')


    def play_step(self, seed):
        self.cells = seed

        if (self.running == False):
            screen.fill(COLOR_GRID)
            self.num_states = 0
            self.running = True
            self.log.clear()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.cells = self.update_ui(screen, self.size, with_progress=True)
        self.num_states += 1

        # turn grid into numeric hash
        coded = self.cells_to_string()
        # prevent infinite loops
        if any(coded == save for save in self.log):
           self.running = False
        self.log.append(coded)

        pygame.display.update()
        time.sleep(0.0005)
        return self.cells, self.num_states, self.running
    
    def reset(self, new_cells):
        self.cells = new_cells
        pass

    def cells_to_string(self):
        # is this too inefficient ??
        s = ''
        for row in self.cells:
            for col in row:
                s += str(col)

        return str(int(s, 2))
        
    def update_ui(self, screen, size, with_progress):
        updated_cells = np.zeros((self.cells.shape[0], self.cells.shape[1]), dtype=np.int8)

        for row, col in np.ndindex(self.cells.shape):
            alive = np.sum(self.cells[row-1:row+2, col-1:col+2]) - self.cells[row, col]
            color = COLOR_BG if self.cells[row, col] == 0 else COLOR_ALIVE_NEXT

            if self.cells[row, col] == 1:
                if alive < 2 or alive > 3:
                    if with_progress:
                        color = COLOR_BG
                elif 2 <= alive <= 3:
                    updated_cells[row, col] = 1
                    if with_progress:
                        color = COLOR_ALIVE_NEXT

            else:
                if alive == 3:
                    updated_cells[row, col] = 1
                    if with_progress:
                        color = COLOR_ALIVE_NEXT
            
            pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))

        return updated_cells
    
    def show_diff(self, predicted, loop, state):
        size = self.size
        updated_cells = np.zeros((self.cells.shape[0], self.cells.shape[1]), dtype=np.int8)

        for row, col in np.ndindex(self.cells.shape):
            

            if self.cells[row, col] == 0 and predicted[row, col] > 0:
                r = min(int(200 * abs(self.cells[row, col] - predicted[row, col])) + 50, 250)
                color = (r, 0, 0)
            elif self.cells[row, col] == 1 and predicted[row, col] < 0:
                color = COLOR_ALIVE_NEXT
            elif self.cells[row, col] == 1 and self.cells[row, col] > 0:
                g = min(int(100 * abs(self.cells[row, col] - predicted[row, col])), 250)
                color = (g, 255, g)
            else:
                color = COLOR_BG
            pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))
            text = round(predicted[row, col].item(), 2)
            text = str(text)
            self.draw_text(text, (col * size, row * size, size - 1, size - 1))
        
        pygame.display.update()
        rect = pygame.Rect(0, 0, WIDTH * 20, LENGTH * 20)
        sub = screen.subsurface(rect)
        pygame.image.save(sub, f'screen\screenshot{loop}_{state}.jpg')
 

    def draw_text(self, num, coords):
        font = pygame.font.SysFont('arial', 9)
        img = font.render(num, False, (255, 166, 241))
        screen.blit(img, coords)
    
    def run_simulation(self):
        pass

