import env
from env import State

from collections import deque
import numpy as np

GAMMA = 1.0
EPSILON = 0.1
ALPHA = 0.1
LAMBDA = 0.5
HISTORY_SIZE = 9


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


def update_value(v: list[float], past_states: deque[State], next_s: State):
    td_error = -env.r(next_s) - GAMMA * v[next_s.encode()] - v[past_states[-1].encode()]
    gl = GAMMA * LAMBDA
    eligibility = 1.0
    for s in reversed(past_states):
        v[s.encode()] += ALPHA * eligibility * td_error
        # 手番が変わるごとに適格度の反転と減衰
        eligibility *= -gl  
        

def exec_episode(s: State, v: list[float]):
    """
    エピソードを1回実行する. 1ステップごとに状態価値関数の値も更新する.
    """
    past_states = deque(maxlen=HISTORY_SIZE)
    while not env.is_terminal(s):
        past_states.append(s)
        pi = epsilon_greedy_policy(s, v)
        a = np.random.choice(list(env.action_space(s)), p=pi)
        next_s = env.f(s, a)
        update_value(v, past_states, next_s)
        s = next_s


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
    # 状態価値関数テーブル
    v = [0.0] * env.STATE_SPACE_SIZE

    NUM_SIMULATIONS = 100000
    for _ in range(NUM_SIMULATIONS):
        exec_episode(State(), v)

    best_pv(v)

main()