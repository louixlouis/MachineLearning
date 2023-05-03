import pygame
import random

# 게임 화면 크기
WIDTH = 800
HEIGHT = 600

# 색깔 상수
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# 알의 크기
EGG_SIZE = 50

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("알까기 게임")

clock = pygame.time.Clock()

class Egg(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([EGG_SIZE, EGG_SIZE])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH - EGG_SIZE)
        self.rect.y = random.randrange(HEIGHT - EGG_SIZE)

    def update(self):
        self.rect.y += 5
        if self.rect.y > HEIGHT:
            self.rect.x = random.randrange(WIDTH - EGG_SIZE)
            self.rect.y = random.randrange(-100, -EGG_SIZE)

all_sprites = pygame.sprite.Group()
eggs = pygame.sprite.Group()

for i in range(10):
    egg = Egg()
    all_sprites.add(egg)
    eggs.add(egg)

score = 0

font = pygame.font.Font(None, 36)

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for egg in eggs:
                if egg.rect.collidepoint(pos):
                    egg.rect.x = random.randrange(WIDTH - EGG_SIZE)
                    egg.rect.y = random.randrange(-100, -EGG_SIZE)
                    score += 1

    all_sprites.update()

    screen.fill(WHITE)

    all_sprites.draw(screen)

    score_text = font.render("Score: " + str(score), True, BLACK)
    screen.blit(score_text, [10, 10])

    pygame.display.flip()

    clock.tick(60)

pygame.quit()