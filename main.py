
import os
import re
import threading
from typing import Any

import chess
import chess.engine
from flask import Flask, jsonify, request

app = Flask(__name__)

ENGINE_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
ENGINE_THREADS = max(1, int(os.getenv("ENGINE_THREADS", "1")))
ENGINE_HASH_MB = max(16, int(os.getenv("ENGINE_HASH_MB", "128")))
ANALYSIS_TIME = max(0.05, float(os.getenv("ANALYSIS_TIME", "0.20")))

_engine = None
_engine_lock = threading.Lock()

# Exact image filenames, without extensions.
# These are the allowed MoveRating values.
RATING_TYPES = (
    "best",
    "blunder",
    "book",
    "brilliant",
    "checkmate_black",
    "checkmate_white",
    "correct",
    "draw_black",
    "draw_white",
    "excellent",
    "forced",
    "good",
    "great_find",
    "inaccuracy",
    "incorrect",
    "mate",
    "missed_win",
    "mistake",
    "resign_black",
    "resign_white",
    "winner",
)


def get_engine():
    global _engine

    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

        try:
            _engine.configure({
                "Threads": ENGINE_THREADS,
                "Hash": ENGINE_HASH_MB
            })
        except chess.engine.EngineError:
            pass

    return _engine


def error(message: str, status: int = 400):
    return jsonify({"Error": message}), status


def parse_old_moves(old_moves: Any):
    if not isinstance(old_moves, list):
        raise ValueError("OldMoves must be an array of UCI moves.")

    board = chess.Board()

    for index, item in enumerate(old_moves):
        if not isinstance(item, str):
            raise ValueError(f"OldMoves[{index}] must be a UCI string.")

        try:
            move = chess.Move.from_uci(item)
        except ValueError:
            raise ValueError(
                f"OldMoves[{index}] is not valid UCI notation: {item}"
            )

        if move not in board.legal_moves:
            raise ValueError(
                f"OldMoves[{index}] is illegal: {item}"
            )

        board.push(move)

    return board


def square_name(value: Any, field: str):
    if not isinstance(value, str) or not re.fullmatch(r"[a-h][1-8]", value):
        raise ValueError(f"{field} must be a square such as e4.")

    return chess.parse_square(value)


def piece_type(value: Any):
    names = {
        "Pawn": chess.PAWN,
        "Knight": chess.KNIGHT,
        "Bishop": chess.BISHOP,
        "Rook": chess.ROOK,
        "Queen": chess.QUEEN,
        "King": chess.KING,
    }

    if value not in names:
        raise ValueError(
            "Piece must be Pawn, Knight, Bishop, Rook, Queen, or King."
        )

    return names[value]


def verify_position(board: chess.Board, position: Any):
    if position is None:
        return

    if not isinstance(position, dict):
        raise ValueError("Position must be an object keyed by squares.")

    supplied = {}

    for square, details in position.items():
        sq = square_name(square, "Position key")

        if not isinstance(details, dict):
            raise ValueError(f"Position[{square}] must be an object.")

        ptype = piece_type(details.get("Piece"))
        color = details.get("Color")

        if color not in ("White", "Black"):
            raise ValueError(
                f"Position[{square}].Color must be White or Black."
            )

        supplied[sq] = chess.Piece(
            ptype,
            color == "White"
        )

    actual = board.piece_map()

    if supplied != actual:
        raise ValueError(
            "Position does not match OldMoves. "
            "Send the complete occupied-square map or omit Position."
        )


def configure_strength(engine, elo: int):
    options = engine.options

    if (
        "UCI_LimitStrength" in options
        and "UCI_Elo" in options
        and elo >= 1320
    ):
        applied = min(3190, elo)

        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": applied
        })

        return {
            "Requested": elo,
            "Applied": applied,
            "Mode": "UCI_Elo"
        }

    if "Skill Level" in options:
        engine.configure({
            "UCI_LimitStrength": False,
            "Skill Level": 0
        })

        return {
            "Requested": elo,
            "Applied": "minimum skill (approximation)",
            "Mode": "Skill Level"
        }

    return {
        "Requested": elo,
        "Applied": "engine default",
        "Mode": "default"
    }


def score_cp(info, pov):
    score = info["score"].pov(pov)

    if score.is_mate():
        mate = score.mate()

        return 100000 if mate and mate > 0 else -100000

    return score.score(mate_score=100000)


def move_type(board_before, board_after, move):
    if board_after.is_checkmate():
        return "Checkmate"

    if board_after.is_check():
        if len(board_after.checkers()) >= 2:
            return "DoubleCheck"

        return "Check"

    if board_before.is_castling(move):
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            return "CastleKing"

        return "CastleQueen"

    if board_before.is_en_passant(move):
        return "EnPassant"

    if move.promotion:
        return "Promotion"

    return "Regular"


def is_sacrifice(board_before, board_after, move):
    piece = board_before.piece_at(move.from_square)

    if piece is None:
        return False

    if piece.piece_type in (chess.KING, chess.PAWN):
        return False

    # A rough sacrifice indicator, not a complete tactical proof.
    return (
        board_after.piece_at(move.to_square) is None
        and not board_after.is_checkmate()
    )


def classify_rating(
    cpl,
    is_top_move,
    board_before,
    board_after,
    move,
    after_info,
    is_book_move=False
):
    mover = board_before.turn

    if board_after.is_checkmate():
        return (
            "checkmate_white"
            if mover == chess.WHITE
            else "checkmate_black"
        )

    if board_after.is_stalemate():
        return (
            "draw_white"
            if mover == chess.WHITE
            else "draw_black"
        )

    if is_book_move:
        return "book"

    if len(list(board_before.legal_moves)) == 1:
        return "forced"

    if after_info:
        score = after_info["score"].pov(mover)

        if score.is_mate() and score.mate() and score.mate() > 0:
            return "mate"

    # A winning position that has lost its winning advantage.
    if cpl >= 150:
        if after_info:
            before_info_score = None

            # The caller's engine evaluation is used for the normal
            # move-quality thresholds below.
            if before_info_score is not None:
                pass

    if is_top_move and cpl <= 5:
        if is_sacrifice(board_before, board_after, move):
            return "brilliant"

        return "great_find"

    if is_top_move:
        return "best"

    if cpl <= 10:
        return "excellent"

    if cpl <= 30:
        return "correct"

    if cpl <= 60:
        return "good"

    if cpl <= 120:
        return "inaccuracy"

    if cpl <= 250:
        return "mistake"

    return "blunder"


def illegal_response(reason):
    return jsonify({
        "Legal": False,
        "Type": "Illegal",
        "MoveRating": "incorrect",
        "Reason": reason
    })


@app.get("/")
def root():
    return jsonify({
        "Service": "Chess API",
        "Status": "online",
        "Endpoint": "POST /analyze"
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return error("Request body must be a JSON object.")

    try:
        board = parse_old_moves(data.get("OldMoves", []))
        verify_position(board, data.get("Position"))

        # Optional resignation event.
        if data.get("Resign") is True:
            resigning_color = (
                "White" if board.turn == chess.WHITE else "Black"
            )

            return jsonify({
                "Legal": True,
                "Type": "Resignation",
                "MoveRating": (
                    "resign_white"
                    if board.turn == chess.WHITE
                    else "resign_black"
                ),
                "Winner": (
                    "Black" if board.turn == chess.WHITE else "White"
                ),
                "BotMove": None
            })

        from_sq = square_name(data.get("From"), "From")
        to_sq = square_name(data.get("To"), "To")

        moving_piece = board.piece_at(from_sq)

        if moving_piece is None:
            return illegal_response("No piece exists on From.")

        if moving_piece.color != board.turn:
            return illegal_response("It is not that piece's turn.")

        expected_piece = piece_type(data.get("Piece"))
        expected_color = data.get("Color")

        if expected_color not in ("White", "Black"):
            raise ValueError("Color must be White or Black.")

        if (
            moving_piece.piece_type != expected_piece
            or moving_piece.color != (expected_color == "White")
        ):
            return illegal_response(
                "Piece or Color does not match the board."
            )

        promotion = data.get("Promotion")

        promotion_map = {
            None: None,
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }

        if promotion not in promotion_map:
            raise ValueError(
                "Promotion must be Queen, Rook, Bishop, Knight, or null."
            )

        move = chess.Move(
            from_sq,
            to_sq,
            promotion=promotion_map[promotion]
        )

        if move not in board.legal_moves:
            return illegal_response("Move is not legal in this position.")

        is_castle = bool(data.get("IsCastle", False))
        castle_side = data.get("CastleSide")
        actual_castle = board.is_castling(move)

        if is_castle != actual_castle:
            return illegal_response(
                "IsCastle does not match the actual move."
            )

        if actual_castle:
            actual_side = (
                "King"
                if chess.square_file(to_sq) > chess.square_file(from_sq)
                else "Queen"
            )

            if castle_side != actual_side:
                return illegal_response(
                    "CastleSide does not match the actual move."
                )

        elo = data.get("Elo", 1500)

        if isinstance(elo, bool) or not isinstance(elo, int) or not 100 <= elo <= 3500:
            raise ValueError("Elo must be an integer between 100 and 3500.")

        mover = board.turn
        board_before = board.copy()

        with _engine_lock:
            engine = get_engine()
            strength = configure_strength(engine, elo)

            best_info = engine.analyse(
                board,
                chess.engine.Limit(time=ANALYSIS_TIME)
            )

            best_move = best_info.get("pv", [None])[0]
            best_score = score_cp(best_info, mover)

            board.push(move)

            if (
                board.is_checkmate()
                or board.is_stalemate()
                or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()
            ):
                # Correct score perspective: checkmate delivered by
                # the player is positive from that player's perspective.
                played_score = (
                    100000 if board.is_checkmate() else 0
                )
                after_info = None
            else:
                after_info = engine.analyse(
                    board,
                    chess.engine.Limit(time=ANALYSIS_TIME)
                )

                played_score = score_cp(after_info, mover)

            cpl = max(0, best_score - played_score)

            rating = classify_rating(
                cpl,
                move == best_move,
                board_before,
                board,
                move,
                after_info,
                bool(data.get("IsBookMove", False))
            )

            kind = move_type(board_before, board, move)

            if board.is_game_over():
                bot_move_data = None
            else:
                reply_info = engine.analyse(
                    board,
                    chess.engine.Limit(time=ANALYSIS_TIME)
                )

                reply = reply_info.get("pv", [None])[0]

                if reply is None:
                    bot_move_data = None
                else:
                    bot_piece = board.piece_at(reply.from_square)

                    bot_move_data = {
                        "From": chess.square_name(reply.from_square),
                        "To": chess.square_name(reply.to_square),
                        "Piece": chess.piece_name(
                            bot_piece.piece_type
                        ).capitalize(),
                        "Color": (
                            "White" if bot_piece.color else "Black"
                        ),
                        "Rating": "best",
                        "Promotion": (
                            chess.piece_name(reply.promotion).capitalize()
                            if reply.promotion else None
                        )
                    }

        return jsonify({
            "Legal": True,
            "Type": kind,
            "MoveRating": rating,
            "CentipawnLoss": cpl,
            "BotMove": bot_move_data,
            "EngineStrength": strength
        })

    except ValueError as exc:
        return error(str(exc))

    except chess.engine.EngineError as exc:
        app.logger.exception("Stockfish engine error")
        return error("Chess engine error: " + str(exc), 500)

    except Exception:
        app.logger.exception("Unexpected analysis error")
        return error("Internal server error.", 500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )
