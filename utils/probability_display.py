import pygame
import numpy as np


class ProbabilityDisplay:
    def __init__(self, rect: pygame.Rect, bg_color: pygame.Color,
                 bar_width=30, bar_gap=5, bar_max_height=140):
        self.rect = rect
        self.bg_color = bg_color
        self.bar_font = pygame.font.Font(None, 30)
        self.pred_font = pygame.font.Font(None, 40)
        self.bar_width = bar_width
        self.bar_gap = bar_gap
        self.bar_max_height = bar_max_height

    def draw_text(self, screen, text, font, x, y):
        color = (255, 255, 255) - np.array(self.bg_color)
        surface = font.render(text, True, color)
        screen.blit(surface, (x, y))
            
    def draw(self, screen, probas):
        screen.fill(self.bg_color, self.rect)
        highest_bar_idx = np.argmax(probas)
        self.draw_text(screen, f"Prediction: {highest_bar_idx}", self.pred_font, self.rect.x, self.rect.y)

        for i, p in enumerate(probas):
            bar_height = int(p * self.bar_max_height)
            bar_rect = pygame.Rect(
                self.rect.x + i * (self.bar_width + self.bar_gap),
                self.rect.y + self.rect.height - bar_height - self.bar_font.get_height(),
                self.bar_width,
                bar_height
            )
            color = (255, 0, 0) if i == highest_bar_idx else (0, 255, 0)
            pygame.draw.rect(screen, color, bar_rect)
            self.draw_text(
                screen,
                str(i), self.bar_font,
                bar_rect.x + bar_rect.width / 4,
                bar_rect.y + bar_rect.height
            )