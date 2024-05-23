import env
from env import State

import numpy as np

GAMMA = 1.0


def policy(s: State) -> list[float]:
    """
    1手先の自石の数に対応した確率分布に従って着手を行う方策
    """
    counts = []
    for a in env.action_space(s):
        next_s = env.f(s, a)

        # next_sはsよりも1手進んでいるので、State.opponent_disc_countが自分の石数
        counts.append(next_s.opponent_disc_count)

    s = sum(counts)
    return [c / s for c in counts]  # 確率分布に変換


def exec_episode(s: State, policy, visit_counts: list[int]) -> list[(State, float)]:
    """
    エピソードを1回実行し, エピソード中に辿った状態の履歴と報酬を返す.
    """
    history: list[(State, float)] = []
    while not env.is_terminal(s):
        visit_counts[s.encode()] += 1
        a = np.random.choice(list(env.action_space(s)), p=policy(s))
        next_s = env.f(s, a)
        history.append((s, -env.r(next_s)))
        s = next_s

    return history


def eval_policy(v: list[float], visit_counts: list[int], history: list[(State, float)]):
    g = 0.0
    for s, reward in reversed(history): # 履歴を逆向きに辿る
        s_idx = s.encode()
        n = visit_counts[s_idx]
        g += reward
        v[s_idx] += (g - v[s_idx]) / n
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
    visit_counts = [0] * env.STATE_SPACE_SIZE

    NUM_SIMULATIONS = 100000
    for _ in range(NUM_SIMULATIONS):
        histroy = exec_episode(State(), policy, visit_counts)
        eval_policy(v, visit_counts, histroy)

    best_pv(v)

main()

