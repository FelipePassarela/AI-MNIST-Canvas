import pygame
from utils.canvas_display import CanvasDisplay
from utils.probability_display import ProbabilityDisplay
from utils.feature_maps_display import FeatureMapsDisplay
from utils.train_cnn import CNN


CANVAS_WIDTH, CANVAS_HEIGHT = 28, 28
CELL_SCALE = 20
BRUSH_COLOR = (255, 255, 255)
CANVAS_BG_COLOR = (0, 0, 0)

BOTTOM_OFFSET = 200
BAR_WIDTH = 30
BAR_GAP = (CANVAS_WIDTH * CELL_SCALE - 10 * BAR_WIDTH) // 9
BAR_MAX_HEIGHT = 140
PROBAS_BG_COLOR = (255, 255, 255)

PREDICT_INTERVAL = 25
FEATMAPS_WIDTH_OFFSET = 800
FEATMAPS_CMAP = "inferno"


def main():
    pygame.init()
    
    screen = pygame.display.set_mode((
        CANVAS_WIDTH * CELL_SCALE + FEATMAPS_WIDTH_OFFSET, 
        CANVAS_HEIGHT * CELL_SCALE + BOTTOM_OFFSET
    ))
    pygame.display.set_caption("MNIST Canvas")

    model = CNN()
    model.load("models/cnn.pth")

    canvas = CanvasDisplay(CANVAS_WIDTH, CANVAS_HEIGHT, CELL_SCALE, BRUSH_COLOR, CANVAS_BG_COLOR)
    proba_display = ProbabilityDisplay(
        pygame.Rect(0, CANVAS_HEIGHT * CELL_SCALE, CANVAS_WIDTH * CELL_SCALE, BOTTOM_OFFSET),
        PROBAS_BG_COLOR,
        BAR_WIDTH, BAR_GAP, BAR_MAX_HEIGHT
    )
    featmaps_display = FeatureMapsDisplay(
        pygame.Rect(CANVAS_WIDTH * CELL_SCALE, 0, FEATMAPS_WIDTH_OFFSET, CANVAS_HEIGHT * CELL_SCALE + BOTTOM_OFFSET),
        cmap=FEATMAPS_CMAP
    )

    running = True
    drawing = False
    last_pos = None
    last_time_predict = 0
    probas = [0] * 10
    conv1_featmap, conv2_featmap = [], []
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    canvas.clear()
        
        if drawing:
            x, y = pygame.mouse.get_pos()
            x, y = x // CELL_SCALE, y // CELL_SCALE
            if (x, y) != last_pos:
                canvas.paint(x, y)
                last_pos = (x, y)

        current_time = pygame.time.get_ticks()
        if drawing and current_time - last_time_predict > PREDICT_INTERVAL:
            image = canvas.grid
            probas, conv1_featmap, conv2_featmap = model.predict(image)
            last_time_predict = current_time
        
        canvas.draw(screen, 0, 0)
        proba_display.draw(screen, probas)
        featmaps_display.draw(screen, (*conv1_featmap, *conv2_featmap))

        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    main()