import env
from env import State

import numpy as np

GAMMA = 1.0
EPSILON = 0.1
ALPHA = 0.1


def epsilon_greedy_policy(s: State, v: list[float]):
    max_idx = 0
    max_q = -float("inf")
    actions = list(env.action_space(s))
    for i, a in enumerate(actions):
        next_s = env.f(s, a)
        q = -(env.r(next_s) + GAMMA * v[next_s.encode()])
        if q > max_q:
            max_q = q
            max_idx = i

    p = [EPSILON / len(actions) for _ in actions]
    p[max_idx] += 1.0 - EPSILON
    return p


def exec_episode(s: State, v: list[float]) -> list[(State, float)]:
    """
    エピソードを1回実行し, エピソード中に辿った状態の履歴と報酬を返す.
    """
    history: list[(State, float)] = []
    while not env.is_terminal(s):
        pi = epsilon_greedy_policy(s, v)
        a = np.random.choice(list(env.action_space(s)), p=pi)
        next_s = env.f(s, a)
        history.append((s, -env.r(next_s)))
        s = next_s

    return history


def eval_policy(v: list[float], history: list[(State, float)]):
    g = 0.0
    for s, reward in reversed(history): # 履歴を逆向きに辿る
        s_idx = s.encode()
        g += reward
        v[s_idx] += ALPHA * (g - v[s_idx])
        g *= -GAMMA # 手番が切り替わるので収益を反転させる.


def best_pv(v: list[float]):
    """
    vに基づく最善進行を出力する.
    """
    s = State()

    while not env.is_terminal(s):
        print(f"s:\n{s}\n")

        max_q = -float("inf")
        best_a = None
        for a in env.action_space(s):
            next_s = env.f(s, a)
            q = -(env.r(next_s) + GAMMA * v[next_s.encode()])

            print(f"Q(s, {a}) = {q}")

            if q > max_q:
                max_q = q
                best_a = a

        s = env.f(s, best_a)
        print()

    print(f"terminal state:\n{s}")


def main():
    # 状態価値関数テーブルと各状態の訪問回数テーブル
    v = [0.0] * env.STATE_SPACE_SIZE

    NUM_SIMULATIONS = 100000
    for _ in range(NUM_SIMULATIONS):
        histroy = exec_episode(State(), v)
        eval_policy(v, histroy)

    best_pv(v)

main()

