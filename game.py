import time
import pygame
import numpy as np
from collections import deque

pygame.init()
screen = pygame.display.set_mode((800, 600))

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)

class CGOL:

    def __init__(self, w=80, h=60, size=10) -> None:
        self.w = w
        self.h = h
        self.size = size
        self.game_iteration = 0
        self.unique_states = []

        self.cells = np.random.randint(0, 2, (60, 80))
        self.log = deque(maxlen=3)

        screen.fill(COLOR_GRID)
        self.cells = self.update_ui(screen, size, with_progress=True)

        pygame.display.update()



    def play_step(self):

        states = 0
        self.game_iteration += 1
        running = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(COLOR_GRID)

        while running:
            self.cells = self.update_ui(screen, self.size, with_progress=True)
            states += 1
            
            if not any(self.cells == unique_cells for unique_cells in self.unique_states):
                self.cells_to_string()

            if any(np.array_equal(self.cells, save, equal_nan=False) for save in self.log):
                running = False

                        
            self.log.append(self.cells)

            pygame.display.update()

            time.sleep(0.001)
        
        return states, self.game_iteration, len(self.unique_states)
    
    def reset(self, new_cells):
        self.cells = new_cells
        pass

    def cells_to_string(self):
        # is this too inefficient ??
        s = ''
        for row in self.cells:
            for col in row:
                s += str(col)
        self.unique_states.append(s)


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
    
    def run_simulation(self):
        pass


def main():
        
    running = False
    game_iteration = 0
    states = 0
    short_memory = deque([])

    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    (screen, cells, 10)
                    pygame.display.update()

            # if pygame.mouse.get_pressed()[0]:
            #     pos = pygame.mouse.get_pos()
            #     cells[pos[1] // 10, pos[0] // 10] = 1
            #     update(screen, cells, 10)
            #     pygame.display.update()
        
        screen.fill(COLOR_GRID)

        if running:
            cells = update(screen, cells, 10, with_progress=True)
            if any(np.array_equal(cells, save, equal_nan=False) for save in short_memory):
                print(f'states: {states}')
                states = 0
                game_iteration += 1


                # get model.predict
                cells = np.random.randint(0, 2, (60, 80))

            states += 1
            if (len(short_memory) > 3):
                short_memory.popleft()

            short_memory.append(cells)

            pygame.display.update()

        time.sleep(0.001)

