import env
from env import State, Action, BoardCoordinate

GAMMA = 1.0    # 割引率
EPSILON = 1.0e-7    # この値以下の実数値の差は無視する


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


def state_sapce(s: State):
    """
    sから辿れる全ての状態を列挙する
    """
    yield s

    if env.is_terminal(s):
        return

    for a in env.action_space(s):
        for ss in state_sapce(env.f(s, a)):
            yield ss


def eval_policy(policy, v: list[float]):
    converged = True
    for s in state_sapce(State()):
        if env.is_terminal(s):
            # これ以上、報酬を手に入れることはないから
            # 終局している局面の価値は0
            v[s.encode()] = 0.0
            continue

        value = 0.0
        pi = policy(s)
        for a, p in zip(env.action_space(s), pi):
            next_s = env.f(s, a)
            value += p * -(env.r(next_s) + GAMMA * v[next_s.encode()])

        s_idx = s.encode()
        if abs(v[s_idx] - value) < EPSILON:  # 収束している
            converged = converged and True
            continue

        v[s_idx] = value
        converged = False

    return converged


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
    # 各要素に各状態の価値が格納されている
    # 添え字はState.encode()が返す値に対応
    v = [0.0] * env.STATE_SPACE_SIZE

    count = 1
    while True:
        print(f"iteration {count}")
        if eval_policy(policy, v):
            break
        count += 1

    best_pv(v)


main()
