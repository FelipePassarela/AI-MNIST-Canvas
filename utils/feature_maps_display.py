import pygame as pg
import numpy as np
import matplotlib.pyplot as plt


class FeatureMapsDisplay:
    def __init__(self, rect: pg.Rect, cmap="gray"):
        self.rect = rect
        self.cmap = plt.get_cmap(cmap)

        self._grid_size = 8
        self._featmap_width = self.rect.width // self._grid_size
        self._featmap_height = self.rect.height // self._grid_size
        self._positions = [
            (self.rect.x + i * self._featmap_width, self.rect.y + j * self._featmap_height)
            for i in range(self._grid_size) for j in range(self._grid_size)
        ]

    def draw(self, screen, featmaps):
        screen.fill((255, 255, 255), self.rect)
        n_featmaps = len(featmaps)

        for i, pos in enumerate(self._positions):
            if i >= n_featmaps:
                break
            featmap = self.normalize_and_apply_cmap(featmaps[i])
            featmap_surface = pg.surfarray.make_surface(featmap)
            featmap_surface = pg.transform.scale(featmap_surface, (self._featmap_width, self._featmap_height))
            screen.blit(featmap_surface, pos)

    def normalize_and_apply_cmap(self, featmap):
        featmap = np.array(featmap).T
        featmap -= featmap.min()
        featmap /= featmap.max() or 1
        featmap = np.uint8(featmap * 255)
        featmap = self.cmap(featmap)[..., :3] * 255
        return np.uint8(featmap)