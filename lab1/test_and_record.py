import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulation import CityEnv, StrategyCautious

def create_gif():
    env = CityEnv("citymap.txt", spawn_prob=0.03)
    strat = StrategyCautious(env.grid, env.goal_pos)

    # Запускаем со старта #1
    random_start = random.randint(0, 2)
    state = env.reset(start_idx=random_start, seed=None)
    history = []
    done = False

    print("Запись симуляции...")
    while not done and len(history) < 250:
        history.append({
            "agent": state["agent_pos"],
            "vehicles": list(state["vehicle_positions"])
        })
        action = strat.get_action(state)
        state, status, done = env.step(action)

    print(f"Поездка завершена со статусом: {status} за {len(history)} шагов.")

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame):
        ax.clear()
        # Показываем карту: дороги белые, здания черные
        ax.imshow(env.grid == 1, cmap="gray_r")

        # Цель (зеленая звезда)
        gr, gc = env.goal_pos
        ax.plot(gc, gr, "g*", markersize=14, label="Цель")

        snap = history[frame]
        # Машины NPC (красные квадраты)
        if snap["vehicles"]:
            vr, vc = zip(*snap["vehicles"])
            ax.plot(vc, vr, "rs", markersize=6, label="Машины NPC")

        # Агент (синий круг)
        ar, ac = snap["agent"]
        ax.plot(ac, ar, "bo", markersize=10, label="Агент")

        ax.set_title(f"Шаг: {frame} | Машин на карте: {len(snap['vehicles'])}")
        ax.legend(loc="upper right")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100)
    ani.save("agent_simulation.gif", writer="pillow", fps=10)
    print("Готово! Анимация сохранена в 'agent_simulation.gif'. Откройте её и посмотрите!")

if __name__ == "__main__":
    create_gif()