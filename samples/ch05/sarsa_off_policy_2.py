import env
from env import State, Action, BoardCoordinate

import numpy as np

GAMMA = 1.0
EPSILON_START = 0.9
EPSILON_END = 0.1
ALPHA = 0.1


def epsilon_greedy_policy(s: State, q: np.ndarray, epsilon: float):
    s_idx = s.encode()
    actions = list(env.action_space(s))
    max_idx = max(enumerate(q[s_idx, a.coord] for a in actions), key=lambda x: x[1])[0]

    p = [epsilon / len(actions) for _ in actions]
    p[max_idx] += 1.0 - epsilon
    return p


def update_value(q: np.ndarray, s: State, a: Action, next_s: State, next_a: Action, rho: float):
    s_idx = s.encode()
    next_s_idx = next_s.encode()
    td_target = -env.r(next_s) - GAMMA * q[next_s_idx, next_a.coord]
    q[s_idx, a.coord] += ALPHA * (rho * td_target - q[s_idx, a.coord])


def exec_episode(s: State, q: np.ndarray, epsilon: float):
    """
    エピソードを1回実行する. 1ステップごとに行動価値関数の値も更新する.
    """
    b = epsilon_greedy_policy(s, q, epsilon)
    prev_s = s
    prev_a = np.random.choice(list(env.action_space(s)), p=b)
    s = env.f(prev_s, prev_a)
    
    while True:
        if not env.is_terminal(s):
            b = epsilon_greedy_policy(s, q, epsilon)
            pi = epsilon_greedy_policy(s, q, 0.0)
            actions = list(env.action_space(s))
            a_idx = np.random.choice(range(len(actions)), p=b)
            a = actions[a_idx]
            update_value(q, prev_s, prev_a, s, a, pi[a_idx] / b[a_idx])
        else:
            update_value(q, prev_s, prev_a, s, Action(BoardCoordinate.PASS), 1.0)
            break

        prev_s = s
        prev_a = a
        s = env.f(s, a)


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

            if q[s_idx, a.coord] > max_q:
                max_q = q[s_idx, a.coord]
                best_a = a

        s = env.f(s, best_a)
        print()

    print(f"terminal state:\n{s}")


def main():
    # 行動価値関数テーブル
    # 状態数 x 行動数の2次元配列で表現
    # q[s.encode(), a.coord] = q(s, a)
    q = np.zeros(shape=(env.STATE_SPACE_SIZE, env.NUM_SQUARES + 1), dtype=np.float32)

    NUM_SIMULATIONS = 20000
    epsilon_delta = (EPSILON_START - EPSILON_END) / (NUM_SIMULATIONS - 1)
    for i in range(NUM_SIMULATIONS):
        exec_episode(State(), q, EPSILON_START - i * epsilon_delta)

    best_pv(q)

main()