
import os
import random
import threading
import atexit
from typing import Optional

import chess
import chess.engine

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="RO-Stockfish API",
    description="Roblox chess analysis API powered by Stockfish",
    version="3.0.0"
)

STOCKFISH_PATH = os.getenv(
    "STOCKFISH_PATH",
    "/usr/games/stockfish"
)

ANALYSIS_TIME = float(
    os.getenv("ANALYSIS_TIME", "0.5")
)

MAX_MULTIPV = 8
MATE_SCORE = 100000

engine = None
engine_lock = threading.Lock()


# ============================================================
# MODELS
# ============================================================

class SquareData(BaseModel):
    Piece: str
    Color: str


class AnalyzeRequest(BaseModel):

    # Board format:
    # {
    #     "A1": {"Piece": "Rook", "Color": "White"}
    # }

    board: dict[str, SquareData]

    old: str
    new: str

    piece: str
    color: str

    elo: int = Field(
        default=1200,
        ge=250,
        le=3000
    )

    castling: str = "-"
    en_passant: Optional[str] = None

    halfmove: int = Field(default=0, ge=0)
    fullmove: int = Field(default=1, ge=1)

    promotion: Optional[str] = None


# ============================================================
# CONSTANTS
# ============================================================

PIECE_NAMES = {
    "Pawn": chess.PAWN,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Rook": chess.ROOK,
    "Queen": chess.QUEEN,
    "King": chess.KING
}

PROMOTION_PIECES = {
    "Queen": chess.QUEEN,
    "Rook": chess.ROOK,
    "Bishop": chess.BISHOP,
    "Knight": chess.KNIGHT
}


# ============================================================
# ENGINE MANAGEMENT
# ============================================================

def get_engine():

    global engine

    if engine is not None:
        return engine

    if not os.path.isfile(STOCKFISH_PATH):
        raise RuntimeError(
            f"Stockfish executable not found: {STOCKFISH_PATH}"
        )

    engine = chess.engine.SimpleEngine.popen_uci(
        STOCKFISH_PATH
    )

    engine.configure({
        "Threads": 1,
        "Hash": 128
    })

    return engine


def close_engine():

    global engine

    if engine is not None:

        try:
            engine.quit()

        except Exception:
            pass

        engine = None


atexit.register(close_engine)


# ============================================================
# RESPONSE HELPERS
# ============================================================

def error_response(message):

    return {
        "legal": False,
        "classification": None,
        "centipawn_loss": None,
        "best_move": None,
        "recommendation": None,
        "error": message
    }


def move_information(move, board):

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    result = {
        "from": chess.square_name(
            move.from_square
        ).upper(),

        "to": chess.square_name(
            move.to_square
        ).upper(),

        "piece": chess.piece_name(
            piece.piece_type
        ).capitalize(),

        "color": (
            "White" if piece.color else "Black"
        )
    }

    if move.promotion:
        result["promotion"] = chess.piece_name(
            move.promotion
        ).capitalize()

    return result


# ============================================================
# BOARD CREATION
# ============================================================

def create_board(data: AnalyzeRequest):

    board = chess.Board.empty()

    # Add pieces
    for square_name, square_data in data.board.items():

        square_name = square_name.upper()

        if square_name not in chess.SQUARE_NAMES:
            raise ValueError(
                f"Invalid board square: {square_name}"
            )

        if square_data.Piece not in PIECE_NAMES:
            raise ValueError(
                f"Invalid piece: {square_data.Piece}"
            )

        if square_data.Color not in ("White", "Black"):
            raise ValueError(
                f"Invalid color: {square_data.Color}"
            )

        square = chess.parse_square(
            square_name.lower()
        )

        piece_type = PIECE_NAMES[
            square_data.Piece
        ]

        piece_color = (
            square_data.Color == "White"
        )

        board.set_piece_at(
            square,
            chess.Piece(
                piece_type,
                piece_color
            )
        )

    # Side to move
    if data.color not in ("White", "Black"):
        raise ValueError(
            "Color must be White or Black"
        )

    board.turn = data.color == "White"

    # Castling rights
    try:
        board.set_castling_fen(data.castling)

    except ValueError:
        raise ValueError(
            f"Invalid castling rights: {data.castling}"
        )

    # En passant
    if data.en_passant:

        square_name = data.en_passant.upper()

        if square_name not in chess.SQUARE_NAMES:
            raise ValueError(
                f"Invalid en passant square: {square_name}"
            )

        board.ep_square = chess.parse_square(
            square_name.lower()
        )

    # Move counters
    board.halfmove_clock = data.halfmove
    board.fullmove_number = data.fullmove

    # Validate position
    status = board.status()

    if status != chess.STATUS_VALID:

        raise ValueError(
            f"Invalid chess position. Status code: {status}"
        )

    return board


# ============================================================
# MOVE CREATION
# ============================================================

def create_move(data: AnalyzeRequest, board):

    old_square = data.old.upper()
    new_square = data.new.upper()

    if old_square not in chess.SQUARE_NAMES:
        raise ValueError("Invalid starting square")

    if new_square not in chess.SQUARE_NAMES:
        raise ValueError("Invalid destination square")

    from_square = chess.parse_square(
        old_square.lower()
    )

    to_square = chess.parse_square(
        new_square.lower()
    )

    moving_piece = board.piece_at(from_square)

    if moving_piece is None:
        raise ValueError(
            "There is no piece on the starting square"
        )

    if data.piece not in PIECE_NAMES:
        raise ValueError("Invalid piece name")

    if moving_piece.piece_type != PIECE_NAMES[data.piece]:
        raise ValueError(
            "The specified piece does not match the board"
        )

    if data.color not in ("White", "Black"):
        raise ValueError("Invalid moving color")

    expected_color = data.color == "White"

    if moving_piece.color != expected_color:
        raise ValueError(
            "The piece color does not match"
        )

    promotion = None

    if data.promotion:

        if data.promotion not in PROMOTION_PIECES:
            raise ValueError(
                "Invalid promotion piece"
            )

        promotion = PROMOTION_PIECES[
            data.promotion
        ]

    move = chess.Move(
        from_square,
        to_square,
        promotion=promotion
    )

    return move


# ============================================================
# EVALUATION HELPERS
# ============================================================

def score_to_centipawns(score):

    if score.is_mate():

        mate = score.mate()

        if mate is None:
            return 0

        return (
            MATE_SCORE - abs(mate)
            if mate > 0
            else -MATE_SCORE + abs(mate)
        )

    return score.score()


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


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

def choose_recommendation(results, board, elo):

    if not results:
        return None

    candidates = []

    for result in results:

        variation = result.get("pv")

        if not variation:
            continue

        move = variation[0]

        if move not in board.legal_moves:
            continue

        score = result["score"].pov(
            board.turn
        )

        value = score_to_centipawns(score)

        candidates.append({
            "move": move,
            "score": value
        })

    if not candidates:
        return None

    # Higher Elo = more likely to select the best move.
    temperature = max(
        0.05,
        min(
            1.5,
            (2000 - elo) / 1200
        )
    )

    best_score = max(
        candidate["score"]
        for candidate in candidates
    )

    weights = []

    for candidate in candidates:

        difference = (
            best_score - candidate["score"]
        )

        weight = max(
            0.000001,
            min(
                1.0,
                2.71828 ** (
                    -difference / (100 * temperature)
                )
            )
        )

        weights.append(weight)

    selected = random.choices(
        candidates,
        weights=weights,
        k=1
    )[0]

    return selected["move"]


# ============================================================
# STOCKFISH ANALYSIS
# ============================================================

def analyze_position(data: AnalyzeRequest):

    board = create_board(data)

    move = create_move(data, board)

    # Check whether the move is legal.
    if move not in board.legal_moves:

        return error_response(
            "The requested move is illegal"
        )

    # Check engine and analyze safely.
    with engine_lock:

        stockfish = get_engine()

        legal_count = board.legal_moves.count()

        if legal_count == 0:
            return error_response(
                "The position has no legal moves"
            )

        multipv = min(
            MAX_MULTIPV,
            legal_count
        )

        # Analyze original position.
        results = stockfish.analyse(
            board,
            chess.engine.Limit(
                time=ANALYSIS_TIME
            ),
            multipv=multipv
        )

        if isinstance(results, dict):
            results = [results]

        if not results:
            raise RuntimeError(
                "Stockfish returned no analysis"
            )

        # Best engine move.
        best_result = results[0]

        if not best_result.get("pv"):
            raise RuntimeError(
                "Stockfish returned no principal variation"
            )

        best_move = best_result["pv"][0]

        best_score = score_to_centipawns(
            best_result["score"].pov(
                board.turn
            )
        )

        # Evaluate the position after the player's move.
        board_after = board.copy()
        board_after.push(move)

        after_result = stockfish.analyse(
            board_after,
            chess.engine.Limit(
                time=ANALYSIS_TIME
            )
        )

        played_score = score_to_centipawns(
            after_result["score"].pov(
                board_after.turn
            )
        )

        # Convert opponent's score to the original
        # player's perspective.
        played_score = -played_score

        centipawn_loss = max(
            0,
            best_score - played_score
        )

        # Check whether the played move is best.
        is_best = move == best_move

        classification = classify_move(
            centipawn_loss,
            is_best
        )

        # Select recommendation.
        recommendation_move = choose_recommendation(
            results,
            board,
            data.elo
        )

    return {
        "legal": True,

        "classification": classification,

        "centipawn_loss": centipawn_loss,

        "best_move": move_information(
            best_move,
            board
        ),

        "recommendation": move_information(
            recommendation_move,
            board
        ) if recommendation_move else None
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "name": "RO-Stockfish API",
        "version": "3.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "engine": "Stockfish",
        "version": "3.0.0"
    }


@app.post("/analyze")
def analyze(data: AnalyzeRequest):

    try:

        return analyze_position(data)

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except chess.engine.EngineError as error:

        raise HTTPException(
            status_code=503,
            detail=f"Stockfish engine error: {error}"
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Internal analysis error: {error}"
        )
