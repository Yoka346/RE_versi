import env
from env import State, Action, BoardCoordinate

import numpy as np

GAMMA = 1.0
EPSILON = 0.1
ALPHA = 0.1


def epsilon_greedy_policy(s: State, q: np.ndarray, epsilon: float):
    s_idx = s.encode()
    actions = list(env.action_space(s))
    max_idx = max(enumerate(q[s_idx, a.coord] for a in actions), key=lambda x: x[1])[0]

    p = [epsilon / len(actions) for _ in actions]
    p[max_idx] += 1.0 - epsilon
    return p


def update_value(q: np.ndarray, s: State, a: Action, next_s: State):
    s_idx = s.encode()

    if not env.is_terminal(next_s):
        max_q = max(q[next_s.encode(), b.coord] for b in env.action_space(next_s))
        td_target = -(env.r(next_s) + GAMMA * max_q)
    else:
        td_target = -env.r(next_s)

    q[s_idx][a.coord] += ALPHA * (td_target - q[s_idx, a.coord])


def exec_episode(s: State, q: np.ndarray):
    """
    エピソードを1回実行する. 1ステップごとに行動価値関数の値も更新する.
    """
    while not env.is_terminal(s):
        pi = epsilon_greedy_policy(s, q, EPSILON)
        a = np.random.choice(list(env.action_space(s)), p=pi)
        next_s = env.f(s, a)
        update_value(q, s, a, next_s)
        s = next_s


def best_pv(q: np.ndarray):
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
            print(f"Q(s, {a}) = {q[s_idx, a.coord]}")

            if q[s_idx][a.coord] > max_q:
                max_q = q[s_idx, a.coord]
                best_a = a

        s = env.f(s, best_a)
        print()

    print(f"terminal state:\n{s}")


def main():
    # 行動価値関数テーブル
    # 状態数 x 行動数の2次元配列で表現
    # q[s, a.coord] = q(s, a)
    q = np.zeros(shape=(env.STATE_SPACE_SIZE, env.NUM_SQUARES + 1), dtype=np.float32)

    NUM_SIMULATIONS = 100000
    for _ in range(NUM_SIMULATIONS):
        exec_episode(State(), q)

    best_pv(q)

main()

      