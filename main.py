
import os
import random
import shutil
import threading

import chess
import chess.engine

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="RO-Stockfish",
    description="Chess analysis API powered by Stockfish",
    version="2.0.0"
)

STOCKFISH_PATH = os.getenv(
    "STOCKFISH_PATH",
    shutil.which("stockfish") or "/usr/games/stockfish"
)

ANALYSIS_TIME = 0.25
MAX_CANDIDATES = 8

engine = None
engine_lock = threading.Lock()


# ============================================================
# PIECE DATA
# ============================================================

PIECE_TYPES = {
    "Pawn": chess.PAWN,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Rook": chess.ROOK,
    "Queen": chess.QUEEN,
    "King": chess.KING,
}

PROMOTION_TYPES = {
    "Queen": chess.QUEEN,
    "Rook": chess.ROOK,
    "Bishop": chess.BISHOP,
    "Knight": chess.KNIGHT,
}


# ============================================================
# REQUEST MODELS
# ============================================================

class SquareData(BaseModel):
    Piece: str
    Color: str


class AnalyzeRequest(BaseModel):
    board: dict[str, SquareData]

    old: str
    new: str

    piece: str
    color: str

    elo: int = Field(ge=250, le=3000)

    promotion: Optional[str] = None

    castling: str = "-"
    en_passant: Optional[str] = None

    halfmove: int = 0
    fullmove: int = 1


# ============================================================
# STOCKFISH
# ============================================================

def get_engine():
    global engine

    if engine is None:
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


# ============================================================
# BOARD CONSTRUCTION
# ============================================================

def create_board(data: AnalyzeRequest):

    board = chess.Board.empty()

    for square_name, info in data.board.items():

        square_name = square_name.lower()

        if square_name not in chess.SQUARE_NAMES:
            raise ValueError(
                f"Invalid square: {square_name}"
            )

        if info.Piece not in PIECE_TYPES:
            raise ValueError(
                f"Invalid piece: {info.Piece}"
            )

        if info.Color not in ("White", "Black"):
            raise ValueError(
                f"Invalid color: {info.Color}"
            )

        piece = chess.Piece(
            PIECE_TYPES[info.Piece],
            info.Color == "White"
        )

        square = chess.parse_square(square_name)

        board.set_piece_at(square, piece)

    # The submitted player's color is the side to move.
    board.turn = data.color == "White"

    # Castling rights.
    try:
        board.set_castling_fen(data.castling)
    except ValueError:
        raise ValueError("Invalid castling rights.")

    # En passant.
    if data.en_passant:

        ep = data.en_passant.lower()

        if ep not in chess.SQUARE_NAMES:
            raise ValueError("Invalid en passant square.")

        board.ep_square = chess.parse_square(ep)

    board.halfmove_clock = data.halfmove
    board.fullmove_number = data.fullmove

    if not board.is_valid():
        raise ValueError(
            f"Invalid chess position. Status: {board.status()}"
        )

    return board


# ============================================================
# MOVE CONSTRUCTION
# ============================================================

def create_move(data: AnalyzeRequest, board):

    old = data.old.lower()
    new = data.new.lower()

    if old not in chess.SQUARE_NAMES:
        raise ValueError("Invalid starting square.")

    if new not in chess.SQUARE_NAMES:
        raise ValueError("Invalid destination square.")

    if data.piece not in PIECE_TYPES:
        raise ValueError("Invalid submitted piece.")

    if data.color not in ("White", "Black"):
        raise ValueError("Invalid submitted color.")

    from_square = chess.parse_square(old)
    to_square = chess.parse_square(new)

    actual_piece = board.piece_at(from_square)

    if actual_piece is None:
        raise ValueError("There is no piece on the starting square.")

    if actual_piece.piece_type != PIECE_TYPES[data.piece]:
        raise ValueError("Submitted piece does not match the board.")

    if actual_piece.color != (data.color == "White"):
        raise ValueError("Submitted color does not match the board.")

    promotion = None

    if data.promotion:

        if data.promotion not in PROMOTION_TYPES:
            raise ValueError("Invalid promotion piece.")

        promotion = PROMOTION_TYPES[data.promotion]

    move = chess.Move(
        from_square,
        to_square,
        promotion=promotion
    )

    return move


# ============================================================
# MOVE CLASSIFICATION
# ============================================================

def classify_move(centipawn_loss, is_best):

    if is_best:
        return "Best"

    if centipawn_loss <= 30:
        return "Excellent"

    if centipawn_loss <= 70:
        return "Good"

    if centipawn_loss <= 130:
        return "Inaccuracy"

    if centipawn_loss <= 250:
        return "Mistake"

    return "Blunder"


# ============================================================
# ELO-INFLUENCED RECOMMENDATION
# ============================================================

def choose_recommendation(candidates, elo):

    if not candidates:
        return None

    # Higher Elo = stronger preference for the top engine moves.
    temperature = 0.25 + (3000 - elo) / 1000

    weights = []

    for rank, candidate in enumerate(candidates):

        weight = pow(
            2.718281828,
            -rank / temperature
        )

        weights.append(weight)

    selected = random.choices(
        candidates,
        weights=weights,
        k=1
    )[0]

    return selected


# ============================================================
# MOVE INFORMATION
# ============================================================

def move_information(move, board):

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    piece_name = chess.piece_name(
        piece.piece_type
    ).capitalize()

    color_name = (
        "White" if piece.color else "Black"
    )

    result = {
        "from": chess.square_name(move.from_square).upper(),
        "to": chess.square_name(move.to_square).upper(),
        "piece": piece_name,
        "color": color_name
    }

    if move.promotion:
        result["promotion"] = chess.piece_name(
            move.promotion
        ).capitalize()

    return result


# ============================================================
# ANALYSIS
# ============================================================

def analyze_position(data: AnalyzeRequest):

    board = create_board(data)

    move = create_move(data, board)

    # Check whether the submitted move is legal.
    if not board.is_legal(move):

        return {
            "legal": False,
            "reason": "Illegal chess move.",
            "classification": None,
            "centipawn_loss": None,
            "best_move": None,
            "recommendation": None
        }

    with engine_lock:

        stockfish = get_engine()

        # Analyze the current position at full engine strength.
        legal_count = board.legal_moves.count()

        multipv = min(MAX_CANDIDATES, legal_count)

        infos = stockfish.analyse(
            board,
            chess.engine.Limit(time=ANALYSIS_TIME),
            multipv=multipv
        )

        if isinstance(infos, dict):
            infos = [infos]

        candidates = []

        for info in infos:

            if not info.get("pv"):
                continue

            candidate_move = info["pv"][0]

            candidates.append(candidate_move)

        if not candidates:
            raise RuntimeError(
                "Stockfish did not return any candidate moves."
            )

        # The strongest engine move.
        best_move = candidates[0]

        # Evaluate the best position from the moving player's POV.
        best_score = infos[0]["score"].pov(
            board.turn
        ).score(mate_score=100000)

        # Evaluate the position after the submitted move.
        mover = board.turn

        after_board = board.copy()
        after_board.push(move)

        after_info = stockfish.analyse(
            after_board,
            chess.engine.Limit(time=ANALYSIS_TIME)
        )

        played_score = after_info["score"].pov(
            mover
        ).score(mate_score=100000)

        centipawn_loss = max(
            0,
            best_score - played_score
        )

        # Classification is based on full-strength analysis.
        classification = classify_move(
            centipawn_loss,
            move == best_move
        )

        # Recommendation is influenced by the requested Elo.
        recommended_move = choose_recommendation(
            candidates,
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
                recommended_move,
                board
            )
        }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "name": "RO-Stockfish",
        "version": "2.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "online"
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

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
