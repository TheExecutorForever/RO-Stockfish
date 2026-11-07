
import os
import threading
from typing import Optional

import chess
import chess.engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="RO-Stockfish API", version="2.0.0")

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")

engine = None
engine_lock = threading.Lock()


# -------------------------
# MODELS
# -------------------------

class SquareData(BaseModel):
    Piece: str
    Color: str


class AnalyzeRequest(BaseModel):
    board: dict[str, SquareData]

    old: str
    new: str

    piece: str
    color: str

    elo: int = Field(default=1200, ge=250, le=3000)

    castling: str = "-"
    en_passant: Optional[str] = None

    halfmove: int = 0
    fullmove: int = 1

    promotion: Optional[str] = None


# -------------------------
# ENGINE
# -------------------------

def get_engine():
    global engine

    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(
            STOCKFISH_PATH
        )

        engine.configure({
            "Threads": 1,
            "Hash": 128
        })

    return engine


# -------------------------
# BOARD CREATION
# -------------------------

PIECE_NAMES = {
    "Pawn": chess.PAWN,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Rook": chess.ROOK,
    "Queen": chess.QUEEN,
    "King": chess.KING
}


def create_board(data: AnalyzeRequest):

    board = chess.Board.empty()

    for square_name, square_data in data.board.items():

        square_name = square_name.upper()

        if square_name not in chess.SQUARE_NAMES:
            raise ValueError(f"Invalid square: {square_name}")

        if square_data.Piece not in PIECE_NAMES:
            raise ValueError(
                f"Invalid piece: {square_data.Piece}"
            )

        if square_data.Color not in ["White", "Black"]:
            raise ValueError(
                f"Invalid color: {square_data.Color}"
            )

        piece_type = PIECE_NAMES[square_data.Piece]
        color = square_data.Color == "White"

        square = chess.parse_square(
            square_name.lower()
        )

        board.set_piece_at(
            square,
            chess.Piece(piece_type, color)
        )

    board.turn = data.color == "White"

    board.set_castling_fen(data.castling)

    if data.en_passant:
        board.ep_square = chess.parse_square(
            data.en_passant.lower()
        )

    board.halfmove_clock = data.halfmove
    board.fullmove_number = data.fullmove

    if not board.is_valid():
        raise ValueError("Invalid chess position")

    return board


# -------------------------
# MOVE CREATION
# -------------------------

def create_move(data: AnalyzeRequest, board):

    old_square = data.old.lower()
    new_square = data.new.lower()

    if old_square not in chess.SQUARE_NAMES:
        raise ValueError("Invalid old square")

    if new_square not in chess.SQUARE_NAMES:
        raise ValueError("Invalid new square")

    from_square = chess.parse_square(old_square)
    to_square = chess.parse_square(new_square)

    moving_piece = board.piece_at(from_square)

    if moving_piece is None:
        raise ValueError("No piece on old square")

    expected_piece = PIECE_NAMES.get(data.piece)

    if expected_piece is None:
        raise ValueError("Invalid moving piece")

    if moving_piece.piece_type != expected_piece:
        raise ValueError("Piece does not match board")

    expected_color = data.color == "White"

    if moving_piece.color != expected_color:
        raise ValueError("Piece color does not match")

    promotion = None

    if data.promotion:
        promotion = PIECE_NAMES.get(data.promotion)

        if promotion not in [
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT
        ]:
            raise ValueError("Invalid promotion")

    return chess.Move(
        from_square,
        to_square,
        promotion=promotion
    )


# -------------------------
# MOVE CLASSIFICATION
# -------------------------

def classify_move(loss, is_best):

    if is_best:
        return "Best"

    if loss <= 30:
        return "Excellent"

    if loss <= 70:
        return "Good"

    if loss <= 130:
        return "Inaccuracy"

    if loss <= 250:
        return "Mistake"

    return "Blunder"


# -------------------------
# RECOMMENDATION
# -------------------------

def move_information(move, board):

    piece = board.piece_at(move.from_square)

    return {
        "from": chess.square_name(move.from_square).upper(),
        "to": chess.square_name(move.to_square).upper(),
        "piece": chess.piece_name(
            piece.piece_type
        ).capitalize(),
        "color": "White" if piece.color else "Black"
    }


def choose_recommendation(engine, board, elo):

    temperature = max(
        0.05,
        min(1.5, (2000 - elo) / 1200)
    )

    results = engine.analyse(
        board,
        chess.engine.Limit(time=0.25),
        multipv=8
    )

    candidates = []

    for result in results:

        move = result["pv"][0]
        score = result["score"].pov(board.turn)

        if score.is_mate():
            value = 10000 if score.mate() > 0 else -10000
        else:
            value = score.score()

        candidates.append((move, value))

    if not candidates:
        return None

    best_score = max(score for _, score in candidates)

    weights = []

    for move, score in candidates:
        difference = best_score - score
        weight = 2.71828 ** (
            -difference / (100 * temperature)
        )
        weights.append(weight)

    import random

    selected = random.choices(
        candidates,
        weights=weights,
        k=1
    )[0][0]

    return selected


# -------------------------
# ANALYSIS
# -------------------------

def analyze_position(data: AnalyzeRequest):

    board = create_board(data)

    move = create_move(data, board)

    if move not in board.legal_moves:
        return {
            "legal": False,
            "classification": None,
            "centipawn_loss": None,
            "best_move": None,
            "recommendation": None
        }

    with engine_lock:

        stockfish = get_engine()

        # Best move before the player's move
        before = stockfish.analyse(
            board,
            chess.engine.Limit(time=0.5)
        )

        best_move = before["pv"][0]

        best_score = before["score"].pov(board.turn)

        # Evaluate the played move
        board_after = board.copy()
        board_after.push(move)

        after = stockfish.analyse(
            board_after,
            chess.engine.Limit(time=0.5)
        )

        played_score = -after["score"].pov(board.turn)

        if best_score.is_mate() or played_score.is_mate():
            if best_score == played_score:
                loss = 0
            elif best_score.is_mate() and played_score.is_mate():
                loss = 10000
            else:
                loss = 10000
        else:
            loss = max(
                0,
                best_score.score() - played_score.score()
            )

        is_best = move == best_move

        classification = classify_move(
            loss,
            is_best
        )

        recommendation_move = choose_recommendation(
            stockfish,
            board,
            data.elo
        )

    return {
        "legal": True,
        "classification": classification,
        "centipawn_loss": loss,
        "best_move": move_information(best_move, board),
        "recommendation": move_information(
            recommendation_move,
            board
        ) if recommendation_move else None
    }


# -------------------------
# ROUTES
# -------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "RO-Stockfish API",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(data: AnalyzeRequest):

    try:
        return analyze_position(data)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
