
import os
import random

import chess
import chess.engine

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RO-Stockfish")

ENGINE_PATH = os.getenv(
    "STOCKFISH_PATH",
    "/usr/games/stockfish"
)

PIECE_NAMES = {
    "Pawn": chess.PAWN,
    "Rook": chess.ROOK,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Queen": chess.QUEEN,
    "King": chess.KING
}


class Request(BaseModel):
    board: list[list[str]]
    old: str
    new: str
    piece: str
    color: str
    elo: int

    castling: str = "-"
    en_passant: str | None = None
    halfmove: int = 0
    fullmove: int = 1


def create_board(data):
    if len(data.board) != 8:
        raise ValueError("Board needs 8 rows")

    board = chess.Board(None)

    for row_index, row in enumerate(data.board):
        if len(row) != 8:
            raise ValueError("Each row needs 8 squares")

        for col_index, symbol in enumerate(row):
            if symbol == ".":
                continue

            if symbol not in "prnbqkPRNBQK":
                raise ValueError("Invalid piece")

            square = chess.square(
                col_index,
                7 - row_index
            )

            board.set_piece_at(
                square,
                chess.Piece.from_symbol(symbol)
            )

    board.turn = (
        chess.WHITE
        if data.color.lower() == "white"
        else chess.BLACK
    )

    board.set_castling_fen(data.castling)

    board.ep_square = (
        chess.parse_square(data.en_passant)
        if data.en_passant
        else None
    )

    board.halfmove_clock = data.halfmove
    board.fullmove_number = data.fullmove

    if not board.is_valid():
        raise ValueError("Invalid chess position")

    return board


def classify_move(best_score, played_score, move, board):
    loss = best_score - played_score

    if loss <= 0:
        if (
            board.is_capture(move)
            or board.gives_check(move)
            or move.promotion
        ):
            return "Brilliant"

        return "Best"

    if loss <= 20:
        return "Best"

    if loss <= 180:
        return "Good"

    return "Blunder"


def choose_elo_move(infos, elo):
    moves = [
        info["pv"][0]
        for info in infos
        if info.get("pv")
    ]

    if not moves:
        return None

    elo = max(250, min(3250, elo))

    strength = (elo - 250) / 3000
    exponent = 0.2 + 7.8 * strength ** 3

    weights = [
        1 / ((index + 1) ** exponent)
        for index in range(len(moves))
    ]

    return random.choices(
        moves,
        weights=weights,
        k=1
    )[0]


@app.get("/")
def home():
    return {
        "status": "online",
        "name": "RO-Stockfish"
    }


@app.post("/analyze")
def analyze(data: Request):
    try:
        if data.color.lower() not in ("white", "black"):
            raise ValueError("Invalid color")

        if data.piece not in PIECE_NAMES:
            raise ValueError("Invalid piece name")

        if not 250 <= data.elo <= 3250:
            raise ValueError("Elo must be 250-3250")

        board = create_board(data)

        old_square = chess.parse_square(data.old)
        new_square = chess.parse_square(data.new)

        moving_piece = board.piece_at(old_square)

        if moving_piece is None:
            return {
                "legal": False,
                "classification": None,
                "recommendation": None
            }

        expected_color = (
            chess.WHITE
            if data.color.lower() == "white"
            else chess.BLACK
        )

        if (
            moving_piece.color != expected_color
            or moving_piece.piece_type != PIECE_NAMES[data.piece]
        ):
            return {
                "legal": False,
                "classification": None,
                "recommendation": None
            }

        move = chess.Move(old_square, new_square)

        if move not in board.legal_moves:
            return {
                "legal": False,
                "classification": None,
                "recommendation": None
            }

        with chess.engine.SimpleEngine.popen_uci(
            ENGINE_PATH
        ) as engine:

            engine.configure({
                "Threads": 1,
                "Hash": 64
            })

            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=14),
                multipv=min(8, board.legal_moves.count())
            )

            best_score = infos[0]["score"].pov(
                board.turn
            ).score(mate_score=100000)

            best_move = infos[0]["pv"][0]

            after = board.copy()
            after.push(move)

            played_info = engine.analyse(
                after,
                chess.engine.Limit(depth=14)
            )

            played_score = played_info["score"].pov(
                board.turn
            ).score(mate_score=100000)

            classification = classify_move(
                best_score,
                played_score,
                move,
                board
            )

            recommendation = choose_elo_move(
                infos,
                data.elo
            )

            if recommendation is None:
                raise ValueError("No recommendation")

            recommended_piece = board.piece_at(
                recommendation.from_square
            )

            return {
                "legal": True,
                "classification": classification,
                "recommendation": {
                    "from": chess.square_name(
                        recommendation.from_square
                    ),
                    "to": chess.square_name(
                        recommendation.to_square
                    ),
                    "piece": chess.piece_name(
                        recommended_piece.piece_type
                    ).capitalize(),
                    "color": (
                        "White"
                        if recommended_piece.color
                        else "Black"
                    )
                }
            }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
