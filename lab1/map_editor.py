import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_FILE = "map.png"
SIZE = 50

try:
    orig_img = Image.open(IMAGE_FILE).convert("RGB")
except FileNotFoundError:
    print(f"Ошибка: положите файл '{IMAGE_FILE}' в папку с проектом!")
    exit()

# 1 - стены/здания, 0 - дороги
grid = np.ones((SIZE, SIZE), dtype=int)

# Текущий выбранный режим: 'R' (дорога), '2' (старт), '3' (спавн), '4' (цель)
current_mode = 'R'
last_click = []

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(orig_img, extent=[0, SIZE, SIZE, 0])

# Слой дорог
road_overlay = ax.imshow(grid == 0, cmap="Blues", alpha=0.45, extent=[0, SIZE, SIZE, 0], vmin=0, vmax=1)

# Точки для меток
scatter_starts = ax.scatter([], [], c="blue", s=80, label="2: Старт (нужно 3 шт)", zorder=5)
scatter_spawns = ax.scatter([], [], c="red", s=80, marker="s", label="3: Спавн NPC", zorder=5)
scatter_goals = ax.scatter([], [], c="lime", s=140, marker="*", label="4: Доставка (Финиш)", zorder=5)

ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlim(0, SIZE)
ax.set_ylim(SIZE, 0)
ax.grid(color="yellow", linestyle=":", linewidth=0.5, alpha=0.5)


def update_title():
    mode_names = {
        'R': "РЕЖИМ: [R] Рисование дорог (клик 1 -> клик 2)",
        '2': "РЕЖИМ: [2] Установка стартов агента (клик на дорогу)",
        '3': "РЕЖИМ: [3] Установка спавна NPC (клик на дорогу)",
        '4': "РЕЖИМ: [4] Установка цели доставки (клик на дорогу)"
    }
    ax.set_title(f"{mode_names[current_mode]}\n"
                 f"Клавиши переключения: [R]-Дороги, [2]-Старт, [3]-Спавн, [4]-Финиш\n"
                 f"ПКМ — стереть ячейку. Закройте окно для сохранения.", fontsize=10)
    fig.canvas.draw_idle()


def draw_line(c0, r0, c1, r1):
    dx, dy = abs(c1 - c0), abs(r1 - r0)
    sx = 1 if c0 < c1 else -1
    sy = 1 if r0 < r1 else -1
    err = dx - dy
    x, y = c0, r0
    while True:
        if 0 <= x < SIZE and 0 <= y < SIZE:
            grid[y, x] = 0
        if x == c1 and y == r1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def refresh_display():
    road_overlay.set_data(grid == 0)

    # Обновляем координаты маркеров
    s2 = np.argwhere(grid == 2)
    s3 = np.argwhere(grid == 3)
    s4 = np.argwhere(grid == 4)

    scatter_starts.set_offsets(s2[:, [1, 0]] if len(s2) else np.empty((0, 2)))
    scatter_spawns.set_offsets(s3[:, [1, 0]] if len(s3) else np.empty((0, 2)))
    scatter_goals.set_offsets(s4[:, [1, 0]] if len(s4) else np.empty((0, 2)))

    fig.canvas.draw_idle()


def on_click(event):
    global last_click
    if event.xdata is None or event.ydata is None:
        return

    c = int(round(event.xdata))
    r = int(round(event.ydata))
    if not (0 <= c < SIZE and 0 <= r < SIZE):
        return

    # Правый клик — стереть метку или дорогу (вернуть 1)
    if event.button == 3:
        grid[r, c] = 1
        last_click.clear()
        refresh_display()
        return

    if event.button == 1:
        if current_mode == 'R':
            if not last_click:
                last_click.append((c, r))
                ax.plot(c, r, 'ro', markersize=4)
                fig.canvas.draw_idle()
            else:
                c0, r0 = last_click.pop()
                draw_line(c0, r0, c, r)
                refresh_display()

        elif current_mode == '2':
            # Старт ставим на дорогу
            grid[r, c] = 2
            refresh_display()

        elif current_mode == '3':
            # Спавн NPC
            grid[r, c] = 3
            refresh_display()

        elif current_mode == '4':
            # Финиш (цель может быть одна: старые стираем)
            grid[grid == 4] = 0
            grid[r, c] = 4
            refresh_display()


def on_key(event):
    global current_mode, last_click
    key = event.key.upper() if event.key else ''
    if key in ['R', '2', '3', '4']:
        current_mode = key
        last_click.clear()
        update_title()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

update_title()
plt.show()

# --- Проверка перед сохранением ---
count_starts = np.sum(grid == 2)
count_goals = np.sum(grid == 4)
count_spawns = np.sum(grid == 3)

print("\n--- Результаты разметки ---")
print(f"Стартов агента (2): {count_starts} (требуется 3)")
print(f"Точек спавна (3): {count_spawns}")
print(f"Точек цели (4): {count_goals} (требуется 1)")

if count_starts != 3:
    print("ВНИМАНИЕ: По заданию нужно ровно 3 старта агента (код 2)!")
if count_goals < 1:
    print("ВНИМАНИЕ: Не установлена точка доставки (код 4)!")

np.savetxt("citymap.txt", grid, fmt="%d")
print("Файл 'citymap.txt' успешно сохранен с вашими ручными метками!")