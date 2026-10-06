import numpy as np
from collections import deque

def check_connectivity(map_file="citymap.txt"):
    grid = np.loadtxt(map_file, dtype=int)
    starts = list(zip(*np.where(grid == 2)))
    spawns = list(zip(*np.where(grid == 3)))
    goals = list(zip(*np.where(grid == 4)))

    print(f"Размер карты: {grid.shape}")
    print(f"Стартов агента (2): {len(starts)} шт.")
    print(f"Спавнов транспорта (3): {len(spawns)} шт.")
    print(f"Точек цели (4): {len(goals)} шт.")

    if len(starts) != 3:
        print("ВНИМАНИЕ: Требуется ровно 3 старта агента!")
    if len(goals) != 1:
        print("ВНИМАНИЕ: Должна быть ровно 1 цель!")

    goal = goals[0]

    # Проверка достижимости финиша через BFS (поиск в ширину)
    for i, start in enumerate(starts):
        queue = deque([start])
        visited = {start}
        reached = False

        while queue:
            curr = queue.popleft()
            if curr == goal:
                reached = True
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] in (0, 2, 3, 4) and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        if reached:
            print(f"Старт #{i+1} {start} -> Путь к финишу ЕСТЬ!")
        else:
            print(f"ОШИБКА: Из старта #{i+1} {start} финиш НЕДОСТИЖИМ (дорога разорвана)!")

if __name__ == "__main__":
    check_connectivity()