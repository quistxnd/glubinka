import numpy as np
import random
import heapq

ACTIONS = {
    0: (0, 0),   # Остановка (Wait)
    1: (-1, 0),  # Вверх
    2: (1, 0),   # Вниз
    3: (0, -1),  # Влево
    4: (0, 1)    # Вправо
}

class Vehicle:
    def __init__(self, pos, lifetime):
        self.pos = pos
        self.lifetime = lifetime
        self.prev_dir = None

    def choose_move(self, grid, occupied_positions):
        r, c = self.pos
        valid_neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                # Ехать можно только по дорогам (0, 2, 3, 4)
                if grid[nr, nc] in (0, 2, 3, 4):
                    valid_neighbors.append((dr, dc))

        if not valid_neighbors:
            return (0, 0)

        # 1. Тупик (1 сосед): единственный выход — развернуться и выехать
        if len(valid_neighbors) == 1:
            chosen = valid_neighbors[0]

        # 2. Развилка (3 или 4 пути): равновероятно выбирает любое направление (включая обратно)
        elif len(valid_neighbors) >= 3:
            chosen = random.choice(valid_neighbors)

        # 3. Прямая дорога или обычный поворот (2 соседа):
        else:
            opposite_dir = (-self.prev_dir[0], -self.prev_dir[1]) if self.prev_dir else None
            forward_moves = [m for m in valid_neighbors if m != opposite_dir]

            if self.prev_dir in valid_neighbors and random.random() < 0.70:
                # 70% едем прямо
                chosen = self.prev_dir
            elif forward_moves:
                # Поворот по изгибу дороги (без разворота на 180)
                chosen = random.choice(forward_moves)
            else:
                # Разворот только если вперед вообще пути нет
                chosen = valid_neighbors[0]

        # Проверка: если следующая клетка занята другой машиной — останавливаемся (Wait)
        next_pos = (r + chosen[0], c + chosen[1])
        if next_pos in occupied_positions:
            return (0, 0)

        self.prev_dir = chosen
        return chosen

class CityEnv:
    def __init__(self, map_file="citymap.txt", spawn_prob=0.03, max_steps=400):
        self.grid = np.loadtxt(map_file, dtype=int)
        self.spawn_prob = spawn_prob
        self.max_steps = max_steps

        self.start_positions = list(zip(*np.where(self.grid == 2)))
        self.spawn_positions = list(zip(*np.where(self.grid == 3)))
        self.goal_pos = tuple(zip(*np.where(self.grid == 4)))[0]

        self.vehicles = []
        self.agent_pos = None
        self.current_step = 0

    def reset(self, start_idx=0, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent_pos = self.start_positions[start_idx]
        self.vehicles = []
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        return {
            "agent_pos": self.agent_pos,
            "vehicle_positions": [v.pos for v in self.vehicles],
            "goal_pos": self.goal_pos,
            "step": self.current_step
        }

    def step(self, action_idx):
        self.current_step += 1
        done = False
        status = "continue"

        occupied = {v.pos for v in self.vehicles}

        # 1. Спавн NPC: проверяем, что клетка спавна свободна
        for sp in self.spawn_positions:
            if random.random() < self.spawn_prob:
                if sp not in occupied and sp != self.agent_pos:
                    lifetime = random.randint(15, 150)
                    new_v = Vehicle(sp, lifetime)
                    self.vehicles.append(new_v)
                    occupied.add(sp)

        # 2. Движение NPC с учетом занятых клеток
        for v in self.vehicles:
            occupied.discard(v.pos)
            dr, dc = v.choose_move(self.grid, occupied)
            v.pos = (v.pos[0] + dr, v.pos[1] + dc)
            v.lifetime -= 1
            occupied.add(v.pos)

        # Удаление машин с истекшим временем жизни
        self.vehicles = [v for v in self.vehicles if v.lifetime > 0]

        # 3. Перемещение агента
        dr, dc = ACTIONS[action_idx]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
            if self.grid[nr, nc] in (0, 2, 3, 4):
                self.agent_pos = (nr, nc)

        # 4. Проверка условий окончания эпизода
        veh_positions = [v.pos for v in self.vehicles]
        if self.agent_pos in veh_positions:
            done = True
            status = "crash"
        elif self.agent_pos == self.goal_pos:
            done = True
            status = "success"
        elif self.current_step >= self.max_steps:
            done = True
            status = "timeout"

        return self.get_state(), status, done

# --- Алгоритм поиска пути A* ---
def a_star(grid, start, goal, dynamic_obstacles=None):
    if dynamic_obstacles is None:
        dynamic_obstacles = set()

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dr, current[1] + dc)
            if 0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]:
                if grid[next_pos] in (0, 2, 3, 4) and next_pos not in dynamic_obstacles:
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + h(goal, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current

    curr = goal
    path = []
    while curr and curr != start:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path

def pos_to_action(curr, next_step):
    if not next_step or curr == next_step:
        return 0
    dr = next_step[0] - curr[0]
    dc = next_step[1] - curr[1]
    mapping = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
    return mapping.get((dr, dc), 0)

# --- 3 Стратегии агента ---
class StrategyBlind:
    def __init__(self, grid, goal):
        self.grid = grid
        self.goal = goal

    def get_action(self, state):
        curr = state["agent_pos"]
        path = a_star(self.grid, curr, self.goal)
        next_step = path[0] if path else curr
        return pos_to_action(curr, next_step)

class StrategyCautious:
    def __init__(self, grid, goal):
        self.grid = grid
        self.goal = goal

    def get_action(self, state):
        curr = state["agent_pos"]
        veh_positions = set(state["vehicle_positions"])
        path = a_star(self.grid, curr, self.goal)
        if not path:
            return 0
        next_step = path[0]
        if next_step in veh_positions:
            return 0  # Ждем
        return pos_to_action(curr, next_step)

class StrategyDynamicReroute:
    def __init__(self, grid, goal):
        self.grid = grid
        self.goal = goal

    def get_action(self, state):
        curr = state["agent_pos"]
        veh_positions = set(state["vehicle_positions"])
        path = a_star(self.grid, curr, self.goal, dynamic_obstacles=veh_positions)
        if not path:
            return 0
        return pos_to_action(curr, path[0])

if __name__ == "__main__":
    env = CityEnv("citymap.txt")
    print(f"Загружено стартов агента (2): {len(env.start_positions)}")
    print(f"Загружено точек спавна (3): {len(env.spawn_positions)} {env.spawn_positions}")
    print(f"Точка цели (4): {env.goal_pos}")