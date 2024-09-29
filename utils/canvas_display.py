import numpy as np
import pygame as pg


class CanvasDisplay:
    def __init__(self, width, height, cell_scale, brush_color, bg_color):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.cell_scale = cell_scale
        self.brush_color = brush_color
        self.bg_color = bg_color

    def clear(self):
        self.grid.fill(0)

    def paint(self, x, y, brush_size=1.8, max_intensity=1.0):
        y_indices, x_indices = np.indices(self.grid.shape)
        distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        mask = distances <= brush_size
        intensities = max_intensity * (1 - distances / brush_size)
        self.grid[mask] = np.minimum(self.grid[mask] + intensities[mask], max_intensity)

    def draw(self, screen, x, y):
        canvas_rect = pg.Rect(
            x, y, 
            self.width * self.cell_scale, 
            self.height * self.cell_scale
        )
        screen.fill(self.bg_color, canvas_rect)

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    continue

                color = [c * self.grid[i, j] for c in self.brush_color]
                cell_rect = pg.Rect(
                    j * self.cell_scale, 
                    i * self.cell_scale, 
                    self.cell_scale, self.cell_scale)
                pg.draw.rect(screen, color, cell_rect)

    def print(self):
        for row in self.grid:
            print(''.join('X' if cell > 0 else ' ' for cell in row))
        