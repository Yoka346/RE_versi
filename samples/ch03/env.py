"""
4x4リバーシの環境を提供するモジュール
"""

from enum import IntEnum
from typing import Generator

BOARD_SIZE = 4
NUM_SQUARES = BOARD_SIZE ** 2
VALID_MASK = 0xffff

STATE_SPACE_SIZE = 3 ** 16
NUM_SQUARE_STATES = 3


class DiscColor(IntEnum):
    """
    石の色を表す列挙体

    Note
    ----
    DiscColor.NULLは空きマスを表現するのに用いる.
    """
    BLACK = 0
    WHITE = 1
    NULL = 2


def to_opponent_color(color: DiscColor) -> DiscColor:
    return color ^ DiscColor.WHITE


class SquareState(IntEnum):
    """
    マスの状態を表す列挙体
    """
    # 手番側が所有しているマス
    PLAYER = 0

    # 相手側が所有しているマス
    OPPONENT = 1

    # 誰も所有していないマス
    NONE = 2


class BoardCoordinate(IntEnum):
    """
    盤面座標
    """
    A1 = 0
    B1 = 1
    C1 = 2
    D1 = 3

    A2 = 4
    B2 = 5
    C2 = 6
    D2 = 7

    A3 = 8
    B3 = 9
    C3 = 10
    D3 = 11

    A4 = 12
    B4 = 13
    C4 = 14
    D4 = 15

    PASS = 16

    NULL = 17


def parse_coord(s: str) -> BoardCoordinate:
    s = s.lower().strip()
    x, y = ord(s[0]) - 'a', ord(s[1]) - '1'
    if x < 0 or x >= NUM_SQUARES or y < 0 or y >= NUM_SQUARES:
        return BoardCoordinate.NULL
    return BoardCoordinate(x + y * BOARD_SIZE)


class State:
    """
    ゲームの状態(局面)

    Note
    ----
    局面はビットボードというデータ構造で管理する.
    ビットボードでは、石の配置を整数値のビット列で表現する.
    4x4リバーシの場合、黒石と白石それぞれの配置を、32itの整数で表現できる(16マスしかないので).
    """

    def __init__(self, player=0b0000001001000000, opponent=0b0000010000100000, side_to_move=DiscColor.BLACK):
        # 手番側の石の配置
        self.__player = player

        # 相手側の石の配置
        self.__opponent = opponent

        # 手番
        self.__side_to_move = side_to_move

    @property
    def player(self) -> int:
        return self.__player

    @property
    def opponent(self) -> int:
        return self.__opponent

    @property
    def side_to_move(self) -> DiscColor:
        return self.__side_to_move
    
    @property
    def player_disc_count(self) -> int:
        return self.__player.bit_count()
    
    @property
    def opponent_disc_count(self) -> int:
        return self.__opponent.bit_count()

    @property
    def disc_diff(self) -> int:
        return self.__player.bit_count() - self.__opponent.bit_count()

    def encode(self) -> int:
        """
        局面のハッシュ値を返す.

        Note
        ----
        符号化アルゴリズムは、単に盤面を16桁の3進数として解釈し、それを10進数にしたもの.
        """
        p, o = self.player, self.opponent
        code = 0
        mask = 1
        for _ in range(NUM_SQUARES):
            code = code * NUM_SQUARE_STATES
            if o & mask:
                code += SquareState.OPPONENT
            elif not (p & mask):
                code += SquareState.NONE
            mask <<= 1
        return code

    def __str__(self) -> str:
        board_str = ["  A B C D"]
        b, w = (self.player, self.opponent) if self.side_to_move == DiscColor.BLACK else (self.opponent, self.player)
        mask = 1
        for i in range(NUM_SQUARES):
            if i % BOARD_SIZE == 0:
                board_str.append("\n")
                board_str.append(str(i // BOARD_SIZE + 1))
                board_str.append(" ")

            if b & mask:
                board_str.append("X ")
            elif w & mask:
                board_str.append("O ")
            else:
                board_str.append("- ")

            mask <<= 1

        return "".join(board_str)


class Action:
    """
    着手(行動)
    """

    def __init__(self, coord: BoardCoordinate):
        # 着手位置の座標
        self.__coord = coord

    @property
    def coord(self) -> BoardCoordinate:
        return self.__coord

    def __str__(self) -> str:
        return self.__coord.name
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Action):
            return False
        return value.coord == self.coord


def action_space(s: State) -> Generator[Action, None, None]:
    """
    与えられた局面sにおける着手可能位置(行動空間)を求める.
    """
    moves = __calc_moves(s.player, s.opponent)

    if moves.bit_count() == 0:
        if __calc_moves(s.opponent, s.player).bit_count() != 0:
            yield Action(BoardCoordinate.PASS)
        return

    while moves:
        coord = (moves & -moves).bit_length() - 1
        yield Action(BoardCoordinate(coord))
        moves &= moves - 1

def is_terminal(s: State) -> bool:
    """
    終局かどうか.
    """
    p, o = s.player, s.opponent
    return __calc_moves(p, o).bit_count() == 0 and __calc_moves(o, p).bit_count() == 0

def f(s: State, a: Action) -> State:
    """
    状態遷移関数
    """
    if a.coord == BoardCoordinate.PASS:
        return State(s.opponent, s.player, to_opponent_color(s.side_to_move))

    # 裏返る石のビットパターン
    flip = __calc_flip(s.player, s.opponent, a.coord)
    return State(s.opponent ^ flip, s.player | flip | (1 << a.coord), to_opponent_color(s.side_to_move))


def r(s: State) -> int:
    """
    報酬関数

    Note
    ----
    sが終局であれば、sの手番からみた石差を返す.
    """
    return s.player.bit_count() - s.opponent.bit_count() if is_terminal(s) else 0

def __calc_flip(p: int, o: int, coord: BoardCoordinate) -> int:
    """
    裏返る石のビットパターンを計算する.
    """

    flip = 0
    x = 1 << coord

    # 着手位置の隣から連続する相手石を確認.
    masked_o = o & 0x6666
    f = (x << 1) & masked_o
    f |= (f << 1) & masked_o
    f |= (f << 1) & masked_o

    # 連続する相手石が途切れる場所に自石があるかどうか.
    # もしあれば、連続する相手石は裏返る石なのでflipに追加.
    # outflankは包囲するという意味.
    outflank = p & (f << 1)
    if outflank:
        flip |= f

    # 以下, 同様の処理を全方向について行う.
    # 冗長な書き方をしているが、ループで書いても大して速度は変わらない.

    # 右方向
    f = (x >> 1) & masked_o
    f |= (f >> 1) & masked_o
    f |= (f >> 1) & masked_o

    outflank = p & (f >> 1)
    if outflank:
        flip |= f

    # 左上方向
    f = (x << 5) & masked_o
    f |= (f << 5) & masked_o
    f |= (f << 5) & masked_o

    outflank = p & (f << 5)
    if outflank:
        flip |= f

    # 上方向
    f = (x << 4) & o
    f |= (f << 4) & o
    f |= (f << 4) & o

    outflank = p & (f << 4)
    if outflank:
        flip |= f

    # 下方向
    f = (x >> 4) & o
    f |= (f >> 4) & o
    f |= (f >> 4) & o

    outflank = p & (f >> 4)
    if outflank:
        flip |= f

    # 左下方向
    f = (x >> 5) & masked_o
    f |= (f >> 5) & masked_o
    f |= (f >> 5) & masked_o

    outflank = p & (f >> 5)
    if outflank:
        flip |= f

    # 右上方向
    f = (x << 3) & masked_o
    f |= (f << 3) & masked_o
    f |= (f << 3) & masked_o

    outflank = p & (f << 3)
    if outflank:
        flip |= f

    # 右下方向
    f = (x >> 3) & masked_o
    f |= (f >> 3) & masked_o
    f |= (f >> 3) & masked_o

    outflank = p & (f >> 3)
    if outflank:
        flip |= f

    return flip


def __calc_moves(p: int, o: int) -> int:
    """
    着手可能位置のビットパターンを計算する.
    """
    moves = 0

    # 自石の隣から連続する相手石を確認.
    masked_o = o & 0x6666
    f = (p << 1) & masked_o
    f |= (f << 1) & masked_o
    f |= (f << 1) & masked_o

    # この時点では正しい着手可能位置とは限らない.
    # 一番最後に空きマス位置とマスクする必要がある.
    moves |= f << 1

    # 以下, 同様の処理を全方向について行う.

    # 右方向
    f = (p >> 1) & masked_o
    f |= (f >> 1) & masked_o
    f |= (f >> 1) & masked_o
    moves |= f >> 1

    # 左上方向
    f = (p << 5) & masked_o
    f |= (f << 5) & masked_o
    f |= (f << 5) & masked_o
    moves |= f << 5

    # 上方向
    f = (p << 4) & o
    f |= (f << 4) & o
    f |= (f << 4) & o
    moves |= f << 4

    # 下方向
    f = (p >> 4) & o
    f |= (f >> 4) & o
    f |= (f >> 4) & o
    moves |= f >> 4

    # 左下方向
    f = (p >> 5) & masked_o
    f |= (f >> 5) & masked_o
    f |= (f >> 5) & masked_o
    moves |= f >> 5

    # 右上方向
    f = (p << 3) & masked_o
    f |= (f << 3) & masked_o
    f |= (f << 3) & masked_o
    moves |= f << 3

    # 右下方向
    f = (p >> 3) & masked_o
    f |= (f >> 3) & masked_o
    f |= (f >> 3) & masked_o
    moves |= f >> 3

    # 最後に空きマス位置でマスクしたものが着手可能位置
    return moves & (~(p | o) & VALID_MASK)
