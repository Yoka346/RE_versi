import env
from env import State, Action, BoardCoordinate

import random
import numpy as np

GAMMA = 1.0
EPSILON = 0.1
ALPHA = 0.1


def epsilon_greedy_policy(s: State, q: list[np.ndarray], epsilon: float):
    s_idx = s.encode()
    actions = list(env.action_space(s))
    max_idx = max(enumerate(q[0][s_idx, a.coord] + q[1][s_idx, a.coord] for a in actions), key=lambda x: x[1])[0]

    p = [epsilon / len(actions) for _ in actions]
    p[max_idx] += 1.0 - epsilon
    return p


def update_value(q: list[np.ndarray], s: State, a: Action, next_s: State):
    s_idx = s.encode()

    if random.random() < 0.5:
        (q[0], q[1]) = (q[1], q[0])

    if not env.is_terminal(next_s):
        qa = [(q[0][next_s.encode(), b.coord], b) for b in env.action_space(next_s)]
        argmax_a = max(qa, key=lambda x: x[0])[1]
        td_target = -(env.r(next_s) + GAMMA * q[1][next_s.encode(), argmax_a.coord])
    else:
        td_target = -env.r(next_s)

    q[0][s_idx][a.coord] += ALPHA * (td_target - q[0][s_idx, a.coord])


def exec_episode(s: State, q: list[np.ndarray]):
    """
    エピソードを1回実行する. 1ステップごとに行動価値関数の値も更新する.
    """
    while not env.is_terminal(s):
        pi = epsilon_greedy_policy(s, q, EPSILON)
        a = np.random.choice(list(env.action_space(s)), p=pi)
        next_s = env.f(s, a)
        update_value(q, s, a, next_s)
        s = next_s


def best_pv(q: list[np.ndarray]):
    """
    qに基づく最善進行を出力する.
    """
    s = State()

    while not env.is_terminal(s):
        print(f"s:\n{s}\n")

        max_q = -float("inf")
        best_a = None
        s_idx = s.encode()
        for a in env.action_space(s):
            value = (q[0][s_idx, a.coord] + q[1][s_idx, a.coord]) * 0.5
            print(f"Q(s, {a}) = {value}")

            if value > max_q:
                max_q = value
                best_a = a

        s = env.f(s, best_a)
        print()

    print(f"terminal state:\n{s}")


def main():
    # 行動価値関数テーブル
    # 2つのQテーブルをリストで保持
    q = [np.zeros(shape=(env.STATE_SPACE_SIZE, env.NUM_SQUARES + 1), dtype=np.float32) for _ in range(2)]

    NUM_SIMULATIONS = 100000
    for _ in range(NUM_SIMULATIONS):
        exec_episode(State(), q)

    best_pv(q)

main()

      