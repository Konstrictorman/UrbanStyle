import pygame

class Button:
    def __init__(self, x, y, width, height, text, inactive_color, active_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.action = action
        self.font = pygame.font.Font(None, 30) # You can customize the font

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        current_color = self.inactive_color

        if self.rect.collidepoint(mouse_pos):
            current_color = self.active_color

        pygame.draw.rect(screen, current_color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0)) # Black text
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()

#Example usage in a game loop:
pygame.init()
screen = pygame.display.set_mode((800, 600))

def quit_game():
    pygame.quit()
    sys.exit()

start_button = Button(300, 200, 200, 50, "Start Game", (0, 200, 0), (0, 255, 0))
quit_button = Button(300, 300, 200, 50, "Quit", (200, 0, 0), (255, 0, 0), quit_game)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        start_button.handle_event(event)
        quit_button.handle_event(event)

    screen.fill((255, 255, 255)) # White background
    start_button.draw(screen)
    quit_button.draw(screen)
    pygame.display.flip()

pygame.quit()