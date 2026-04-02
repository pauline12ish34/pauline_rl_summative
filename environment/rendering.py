#!/usr/bin/env python3
"""
Visualization GUI components for warehouse environment
"""
import pygame
import math

def render_warehouse(env, mode="human"):
    """Render the warehouse environment with enhanced visuals. Supports 'human' and 'rgb_array' modes."""
    import numpy as np

    cell_size = 80
    width = env.grid_size * cell_size + 250
    height = env.grid_size * cell_size + 100

    # Modern harmonious color palette
    TILE_COLOR = (120, 98, 61)        # #78623D dark beige/brown
    GRID_LINE = (80, 70, 50)          # #504632 dark brown/gray
    ROBOT_MAIN = (0, 119, 182)        # #0077B6 deep teal
    ROBOT_ACCENT = (144, 224, 239)    # #90E0EF light teal
    OBSTACLE = (255, 111, 97)         # #FF6F61 coral red
    ITEM = (255, 215, 0)              # #FFD700 gold
    TARGET = (46, 196, 182)           # #2EC4B6 emerald green
    TARGET_GLOW = (203, 243, 240)     # #CBF3F0 soft glow
    UI_BG = (247, 247, 255)           # #F7F7FF off-white
    TEXT = (34, 34, 59)               # #22223B charcoal
    HIGHLIGHT = (46, 196, 182)        # #2EC4B6 emerald
    WHITE = (255, 255, 255)


    if mode == "human":
        if not hasattr(env, 'window') or env.window is None:
            pygame.init()
            pygame.display.init()
            try:
                env.window = pygame.display.set_mode((width, height))
                pygame.display.set_caption("🏥 Vaccine Cold-Chain Storage Robot - Rwanda")
                env.clock = pygame.time.Clock()
            except pygame.error:
                return None
        surface = env.window
    else:
        pygame.init()
        surface = pygame.Surface((width, height))


    # Fill background
    surface.fill(TILE_COLOR)

    # Draw grid with modern colors
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, TILE_COLOR, rect)
            pygame.draw.rect(surface, GRID_LINE, rect, 2)

    # ...existing code for obstacles, items, targets, robot, UI panel...
    # Draw obstacles
    for obs in env.obstacles:
        center_x = obs[0] * cell_size + cell_size // 2
        center_y = obs[1] * cell_size + cell_size // 2
        size = int(cell_size * 0.8)
        obstacle_rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
        pygame.draw.rect(surface, OBSTACLE, obstacle_rect)
        pygame.draw.rect(surface, WHITE, obstacle_rect, 3)

    # Draw items
    for item in env.items:
        center_x = item[0] * cell_size + cell_size // 2
        center_y = item[1] * cell_size + cell_size // 2
        pygame.draw.circle(surface, ITEM, (center_x, center_y), cell_size // 3)
        pygame.draw.circle(surface, WHITE, (center_x, center_y), cell_size // 4)

    # Draw targets
    for target in env.targets:
        center_x = target[0] * cell_size + cell_size // 2
        center_y = target[1] * cell_size + cell_size // 2
        for ring in range(3):
            radius = cell_size // 3 + ring * 5
            color = TARGET_GLOW if ring == 0 else TARGET
            pygame.draw.circle(surface, color, (center_x, center_y), radius, 3)

    # Draw robot
    robot_center_x = env.robot_pos[0] * cell_size + cell_size // 2
    robot_center_y = env.robot_pos[1] * cell_size + cell_size // 2
    pygame.draw.circle(surface, ROBOT_MAIN, (robot_center_x, robot_center_y), cell_size // 2)
    pygame.draw.circle(surface, ROBOT_ACCENT, (robot_center_x, robot_center_y), cell_size // 3)
    pygame.draw.circle(surface, WHITE, (robot_center_x - 8, robot_center_y - 8), cell_size // 6)

    # Vaccine carrying indicator
    if env.carrying > 0:
        for i in range(env.carrying):
            carry_x = robot_center_x + (i - env.carrying/2) * 15
            carry_y = robot_center_y - 25
            pygame.draw.circle(surface, WHITE, (int(carry_x), int(carry_y)), 8)
            pygame.draw.circle(surface, HIGHLIGHT, (int(carry_x), int(carry_y)), 5)

    # UI Panel
    ui_x = env.grid_size * cell_size + 20
    font_large = pygame.font.Font(None, 32)
    font_medium = pygame.font.Font(None, 24)
    title = font_large.render("🏥 VACCINE ROBOT", True, WHITE)
    surface.blit(title, (ui_x, 20))
    subtitle = font_medium.render("Rwanda Health", True, HIGHLIGHT)
    surface.blit(subtitle, (ui_x, 50))
    stats = [
        f"❄️ Cold-Chain: {env.steps}/{env.max_steps}",
        f"💉 Vaccines: {env.carrying}/{env.max_inventory}",
        f"🏥 Delivered: {env.delivered_items}",
        f"📦 Boxes Left: {len(env.items)}",
        f"❄️ Status: {'LOADED' if env.carrying else 'READY'}"
    ]
    for i, stat in enumerate(stats):
        color = HIGHLIGHT if "READY" in stat or "LOADED" in stat else WHITE
        text = font_medium.render(stat, True, color)
        surface.blit(text, (ui_x, 85 + i * 30))
    # Cold-Chain Timer Progress bar
    progress = (env.max_steps - env.steps) / env.max_steps
    bar_width = 180
    bar_height = 20
    timer_label = font_medium.render("❄️ Cold-Chain Timer", True, WHITE)
    surface.blit(timer_label, (ui_x, 215))
    pygame.draw.rect(surface, (50, 50, 50), (ui_x, 240, bar_width, bar_height))
    fill_width = int(bar_width * progress)
    color = HIGHLIGHT if progress > 0.5 else ITEM if progress > 0.2 else OBSTACLE
    pygame.draw.rect(surface, color, (ui_x, 240, fill_width, bar_height))
    pygame.draw.rect(surface, WHITE, (ui_x, 240, bar_width, bar_height), 2)

    if mode == "human":
        pygame.display.flip()
        env.clock.tick(8)
        return None
    elif mode == "rgb_array":
        # Convert surface to RGB array
        rgb_array = pygame.surfarray.array3d(surface)
        # Pygame's array shape is (width, height, 3), transpose to (height, width, 3)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        return rgb_array
    
    # Draw grid
    for x in range(env.grid_size + 1):
        for y in range(env.grid_size + 1):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(env.window, SILVER, rect, 2)
    
    # Draw obstacles
    for obs in env.obstacles:
        center_x = obs[0] * cell_size + cell_size // 2
        center_y = obs[1] * cell_size + cell_size // 2
        size = int(cell_size * 0.8)
        
        obstacle_rect = pygame.Rect(center_x - size//2, center_y - size//2, size, size)
        pygame.draw.rect(env.window, RUBY, obstacle_rect)
        pygame.draw.rect(env.window, WHITE, obstacle_rect, 3)
    
    # Draw items
    for item in env.items:
        center_x = item[0] * cell_size + cell_size // 2
        center_y = item[1] * cell_size + cell_size // 2
        
        pygame.draw.circle(env.window, GOLD, (center_x, center_y), cell_size // 3)
        pygame.draw.circle(env.window, WHITE, (center_x, center_y), cell_size // 4)
    
    # Draw targets
    for target in env.targets:
        center_x = target[0] * cell_size + cell_size // 2
        center_y = target[1] * cell_size + cell_size // 2
        
        for ring in range(3):
            radius = cell_size // 3 + ring * 5
            pygame.draw.circle(env.window, EMERALD, (center_x, center_y), radius, 3)
    
    # Draw robot
    robot_center_x = env.robot_pos[0] * cell_size + cell_size // 2
    robot_center_y = env.robot_pos[1] * cell_size + cell_size // 2
    
    pygame.draw.circle(env.window, ROBOT_BLUE, (robot_center_x, robot_center_y), cell_size // 2)
    pygame.draw.circle(env.window, LIGHT_BLUE, (robot_center_x, robot_center_y), cell_size // 3)
    pygame.draw.circle(env.window, WHITE, (robot_center_x - 8, robot_center_y - 8), cell_size // 6)
    
    # Vaccine carrying indicator
    if env.carrying > 0:
        for i in range(env.carrying):
            carry_x = robot_center_x + (i - env.carrying/2) * 15
            carry_y = robot_center_y - 25
            pygame.draw.circle(env.window, WHITE, (int(carry_x), int(carry_y)), 8)
            pygame.draw.circle(env.window, EMERALD, (int(carry_x), int(carry_y)), 5)
    
    # UI Panel
    ui_x = env.grid_size * cell_size + 20
    font_large = pygame.font.Font(None, 32)
    font_medium = pygame.font.Font(None, 24)
    
    title = font_large.render("🏥 VACCINE ROBOT", True, WHITE)
    env.window.blit(title, (ui_x, 20))
    
    subtitle = font_medium.render("Rwanda Health", True, EMERALD)
    env.window.blit(subtitle, (ui_x, 50))
    
    stats = [
        f"❄️ Cold-Chain: {env.steps}/{env.max_steps}",
        f"💉 Vaccines: {env.carrying}/{env.max_inventory}",
        f"🏥 Delivered: {env.delivered_items}",
        f"📦 Boxes Left: {len(env.items)}",
        f"❄️ Status: {'LOADED' if env.carrying else 'READY'}"
    ]
    
    for i, stat in enumerate(stats):
        color = EMERALD if "READY" in stat or "LOADED" in stat else WHITE
        text = font_medium.render(stat, True, color)
        env.window.blit(text, (ui_x, 85 + i * 30))
    
    # Cold-Chain Timer Progress bar
    progress = (env.max_steps - env.steps) / env.max_steps
    bar_width = 180
    bar_height = 20
    
    timer_label = font_medium.render("❄️ Cold-Chain Timer", True, WHITE)
    env.window.blit(timer_label, (ui_x, 215))
    
    pygame.draw.rect(env.window, (50, 50, 50), (ui_x, 240, bar_width, bar_height))
    
    fill_width = int(bar_width * progress)
    color = EMERALD if progress > 0.5 else GOLD if progress > 0.2 else RUBY
    pygame.draw.rect(env.window, color, (ui_x, 240, fill_width, bar_height))
    
    pygame.draw.rect(env.window, WHITE, (ui_x, 240, bar_width, bar_height), 2)
    
    pygame.display.flip()
    env.clock.tick(8)
    
    return None