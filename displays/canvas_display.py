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
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        min_x = max(int(x - brush_size), 0)
        max_x = min(int(x + brush_size), self.width - 1)
        min_y = max(int(y - brush_size), 0)
        max_y = min(int(y + brush_size), self.height - 1)

        y_indices, x_indices = np.indices((max_y - min_y + 1, max_x - min_x + 1))
        y_indices += min_y
        x_indices += min_x

        distances = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        mask = distances <= brush_size
        intensity = max_intensity * (1 - distances / brush_size)

        self.grid[min_y:max_y + 1, min_x:max_x + 1][mask] = np.minimum(
            self.grid[min_y:max_y + 1, min_x:max_x + 1][mask] + intensity[mask], 1.0
        )

    def draw(self, screen, x, y):
        mask = self.grid > 0
        color_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        color_array[mask] = (self.grid[mask][:, None] * self.brush_color).astype(np.uint8)
        color_array = np.transpose(color_array, (1, 0, 2))

        surf = pg.surfarray.make_surface(color_array)
        scaled_surf = pg.transform.scale(surf, (self.width * self.cell_scale, self.height * self.cell_scale))
        screen.blit(scaled_surf, (x, y))

    def print(self):
        for row in self.grid:
            print(''.join('X' if cell > 0 else ' ' for cell in row))
        