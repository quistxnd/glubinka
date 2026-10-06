import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import CityEnv, StrategyBlind, StrategyCautious, StrategyDynamicReroute


def run_experiments():
    print("=== 1/3. Запуск серийного моделирования ===")
    env = CityEnv("citymap.txt")
    strategies = {
        "Blind (Слепой A*)": StrategyBlind(env.grid, env.goal_pos),
        "Cautious (Ожидание)": StrategyCautious(env.grid, env.goal_pos),
        "Dynamic (Объезд)": StrategyDynamicReroute(env.grid, env.goal_pos)
    }

    seeds = [42, 105, 333, 777, 999]  # 5 разных random_state по ТЗ
    spawn_probs = [0.01, 0.03, 0.05, 0.08]  # Разные скорости генерации трафика
    runs_per_config = 100  # 100 экспериментов на конфигурацию

    records = []

    for p in spawn_probs:
        env.spawn_prob = p
        print(f"Тестирование плотности трафика p = {p}...")
        for strat_name, strat in strategies.items():
            for start_idx in range(len(env.start_positions)):
                for seed in seeds:
                    for r_id in range(runs_per_config):
                        ep_seed = seed * 1000 + r_id
                        state = env.reset(start_idx=start_idx, seed=ep_seed)
                        done = False

                        while not done:
                            action = strat.get_action(state)
                            state, status, done = env.step(action)

                        records.append({
                            "spawn_prob": p,
                            "strategy": strat_name,
                            "start_idx": start_idx + 1,
                            "seed": seed,
                            "run_id": r_id,
                            "status": status,
                            "success": 1 if status == "success" else 0,
                            "steps": state["step"]
                        })

    df = pd.DataFrame(records)
    df.to_csv("simulation_results.csv", index=False)
    print("Данные сохранены в simulation_results.csv")
    return df


def generate_report_plots(df):
    print("=== 2/3. Построение графиков для отчета ===")
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. График изменения среднего количества шагов от spawn_prob
    plt.figure(figsize=(8, 5))
    success_df = df[df["status"] == "success"]
    sns.lineplot(
        data=success_df,
        x="spawn_prob",
        y="steps",
        hue="strategy",
        marker="o",
        linewidth=2.2
    )
    plt.title("Зависимость среднего количества шагов от скорости генерации транспорта", fontsize=12)
    plt.xlabel("Вероятность генерации транспорта (spawn_prob)", fontsize=10)
    plt.ylabel("Среднее количество шагов до цели", fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_steps.png", dpi=300)
    plt.close()

    # 2. Boxplot: Скорость доставки (шаги)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=success_df, x="strategy", y="steps", hue="strategy", palette="Set2", legend=False)
    plt.title("Диаграмма размаха (Boxplot) времени доставки успешных агентов", fontsize=12)
    plt.xlabel("Стратегия агента", fontsize=10)
    plt.ylabel("Количество шагов (время доставки)", fontsize=10)
    plt.tight_layout()
    plt.savefig("boxplot_speed.png", dpi=300)
    plt.close()

    # 3. Boxplot: Вероятность успеха агента
    success_rates = df.groupby(["strategy", "spawn_prob", "seed"])["success"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=success_rates, x="strategy", y="success", hue="strategy", palette="Set3", legend=False)
    plt.title("Диаграмма размаха (Boxplot) вероятности успешной доставки", fontsize=12)
    plt.xlabel("Стратегия агента", fontsize=10)
    plt.ylabel("Доля успешных доставок (Success Rate)", fontsize=10)
    plt.tight_layout()
    plt.savefig("boxplot_success.png", dpi=300)
    plt.close()
    print("Графики сохранены: plot_steps.png, boxplot_speed.png, boxplot_success.png")


def capture_frames():
    print("=== 3/3. Создание кадров перемещения агента ===")
    env = CityEnv("citymap.txt", spawn_prob=0.03)
    strat = StrategyCautious(env.grid, env.goal_pos)
    state = env.reset(start_idx=0, seed=42)

    history = []
    done = False
    while not done and len(history) < 200:
        history.append({
            "agent": state["agent_pos"],
            "vehicles": list(state["vehicle_positions"]),
            "step": state["step"]
        })
        action = strat.get_action(state)
        state, status, done = env.step(action)

    # Выбираем 4 контрольные точки пути
    total = len(history)
    selected_indices = [0, total // 3, (2 * total) // 3, total - 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        snap = history[idx]
        ax.imshow(env.grid == 1, cmap="gray_r")
        gr, gc = env.goal_pos
        ax.plot(gc, gr, "g*", markersize=10)
        if snap["vehicles"]:
            vr, vc = zip(*snap["vehicles"])
            ax.plot(vc, vr, "rs", markersize=4)
        ar, ac = snap["agent"]
        ax.plot(ac, ar, "bo", markersize=8)
        ax.set_title(f"Итерация {snap['step']}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("agent_trajectory_frames.png", dpi=300)
    plt.close()
    print("Раскадровка движения сохранена в agent_trajectory_frames.png")


if __name__ == "__main__":
    df = run_experiments()
    generate_report_plots(df)
    capture_frames()
    print("\nВся аналитика готова к вставке в отчёт!")