import pygame as pg
import torch
from displays.canvas_display import CanvasDisplay
from displays.probability_display import ProbabilityDisplay
from displays.feature_maps_display import FeatureMapsDisplay
from mnist_cnn.MnistCNN import MnistCNN


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
FEATMAPS_CMAP = "gray"


def main():
    pg.init()
    
    screen = pg.display.set_mode((
        CANVAS_WIDTH * CELL_SCALE + FEATMAPS_WIDTH_OFFSET, 
        CANVAS_HEIGHT * CELL_SCALE + BOTTOM_OFFSET
    ))
    pg.display.set_caption("AI MNIST Canvas")
    clock = pg.time.Clock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load("models/cnn.pth", map_location=device, weights_only=True))
    model.eval()

    canvas = CanvasDisplay(CANVAS_WIDTH, CANVAS_HEIGHT, CELL_SCALE, BRUSH_COLOR, CANVAS_BG_COLOR)
    proba_display = ProbabilityDisplay(
        pg.Rect(0, CANVAS_HEIGHT * CELL_SCALE, CANVAS_WIDTH * CELL_SCALE, BOTTOM_OFFSET),
        PROBAS_BG_COLOR,
        BAR_WIDTH, BAR_GAP, BAR_MAX_HEIGHT
    )
    featmaps_display = FeatureMapsDisplay(
        pg.Rect(CANVAS_WIDTH * CELL_SCALE, 0, FEATMAPS_WIDTH_OFFSET, CANVAS_HEIGHT * CELL_SCALE + BOTTOM_OFFSET),
        cmap=FEATMAPS_CMAP
    )

    running = True
    drawing = False
    last_pos = None
    last_time_predict = 0
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pg.MOUSEBUTTONUP:
                drawing = False
                last_pos = None
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_c:
                    canvas.clear()
        
        if drawing:
            x, y = pg.mouse.get_pos()
            x, y = x // CELL_SCALE, y // CELL_SCALE
            if (x, y) != last_pos:
                canvas.paint(x, y)
                last_pos = (x, y)

        current_time = pg.time.get_ticks()
        is_time_for_prediction = current_time - last_time_predict > PREDICT_INTERVAL

        if is_time_for_prediction:
            image = canvas.grid
            probas, featmaps = model.predict(image)

            featmaps_display.draw(screen, (*featmaps[0][:32], *featmaps[1][:16], *featmaps[2][:16]))
            proba_display.draw(screen, probas)
            
            last_time_predict = current_time
        
        canvas.draw(screen, 0, 0)
        pg.display.update()

        clock.tick(120)

    pg.quit()


if __name__ == '__main__':
    main()