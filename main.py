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

PIECE_NAMES = {
    "Pawn": chess.PAWN,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Rook": chess.ROOK,
    "Queen": chess.QUEEN,
    "King": chess.KING,
}

RATING_NAMES = {
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
}


def get_engine():
    global _engine

    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

        try:
            _engine.configure({
                "Threads": ENGINE_THREADS,
                "Hash": ENGINE_HASH_MB,
            })
        except chess.engine.EngineError:
            pass

    return _engine


def error(message: str, status: int = 400):
    return jsonify({"Error": message}), status


def illegal(message: str):
    return jsonify({
        "Legal": False,
        "Type": "Illegal",
        "MoveRating": "incorrect",
        "Reason": message,
    })


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
    if value not in PIECE_NAMES:
        raise ValueError(
            "Piece must be Pawn, Knight, Bishop, Rook, Queen, or King."
        )

    return PIECE_NAMES[value]


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

    actual = dict(board.piece_map())

    if supplied != actual:
        raise ValueError(
            "Position does not match OldMoves. "
            "Send every occupied square or omit Position."
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
            "UCI_Elo": applied,
        })

        return {
            "Requested": elo,
            "Applied": applied,
            "Mode": "UCI_Elo",
        }

    if "Skill Level" in options:
        engine.configure({
            "UCI_LimitStrength": False,
            "Skill Level": 0,
        })

        return {
            "Requested": elo,
            "Applied": "minimum skill approximation",
            "Mode": "Skill Level",
        }

    return {
        "Requested": elo,
        "Applied": "engine default",
        "Mode": "default",
    }


def score_cp(info, pov):
    score = info["score"].pov(pov)

    if score.is_mate():
        mate = score.mate()
        return 100000 if mate is not None and mate > 0 else -100000

    return score.score(mate_score=100000)


def has_forced_mate(info, pov):
    if not info:
        return False

    score = info["score"].pov(pov)

    return score.is_mate() and score.mate() is not None and score.mate() > 0


def is_sacrifice(board_after, move, after_info, cpl):
    """
    Conservative sacrifice heuristic.

    The opponent's best reply must capture the moved piece, while
    the engine evaluation must still consider the position sound.
    """

    if after_info is None or cpl > 30:
        return False

    pv = after_info.get("pv", [])

    if not pv:
        return False

    reply = pv[0]

    if not board_after.is_capture(reply):
        return False

    moved_piece = board_after.piece_at(move.to_square)

    if moved_piece is None:
        return False

    captured_piece = board_after.piece_at(reply.to_square)

    if board_after.is_en_passant(reply):
        captured_square = (
            reply.to_square - 8
            if board_after.turn == chess.WHITE
            else reply.to_square + 8
        )
        captured_piece = board_after.piece_at(captured_square)

    if captured_piece is None:
        return False

    sacrificed_value = chess.piece_value(moved_piece.piece_type)
    captured_value = chess.piece_value(captured_piece.piece_type)

    material_cost = sacrificed_value - captured_value

    # Require a meaningful sacrifice, not an ordinary exchange.
    if material_cost < 2:
        return False

    # Do not call a losing blunder a brilliant sacrifice.
    return True


def classify_rating(
    cpl,
    is_top_move,
    forced,
    missed_win,
    brilliant,
    great_find,
):
    if missed_win:
        return "missed_win"

    if brilliant:
        return "brilliant"

    if forced:
        return "forced"

    if great_find:
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


def event_response(rating, event_type):
    return jsonify({
        "Legal": True,
        "Type": event_type,
        "MoveRating": rating,
        "CentipawnLoss": 0,
        "BotMove": None,
    })


@app.get("/")
def root():
    return jsonify({
        "Service": "Chess API",
        "Status": "online",
        "Endpoint": "POST /analyze",
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
        # Game events do not require a chess move.
        game_event = data.get("GameEvent")

        if game_event is not None:
            if game_event == "Winner":
                return event_response("winner", "Winner")

            if game_event == "Resign":
                color = data.get("Color")

                if color not in ("White", "Black"):
                    raise ValueError(
                        "Color must be White or Black for resignation."
                    )

                rating = (
                    "resign_white"
                    if color == "White"
                    else "resign_black"
                )

                return event_response(rating, "Resignation")

            raise ValueError(
                "GameEvent must be null, Winner, or Resign."
            )

        board = parse_old_moves(data.get("OldMoves", []))
        verify_position(board, data.get("Position"))

        from_sq = square_name(data.get("From"), "From")
        to_sq = square_name(data.get("To"), "To")

        moving_piece = board.piece_at(from_sq)

        if moving_piece is None:
            return illegal("No piece exists on From.")

        if moving_piece.color != board.turn:
            return illegal("It is not that piece's turn.")

        expected_piece = piece_type(data.get("Piece"))
        expected_color = data.get("Color")

        if expected_color not in ("White", "Black"):
            raise ValueError("Color must be White or Black.")

        if (
            moving_piece.piece_type != expected_piece
            or moving_piece.color != (expected_color == "White")
        ):
            return illegal("Piece or Color does not match the board.")

        promotion = data.get("Promotion")

        promotion_map = {
            None: None,
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT,
        }

        if promotion not in promotion_map:
            raise ValueError(
                "Promotion must be Queen, Rook, Bishop, Knight, or null."
            )

        move = chess.Move(
            from_sq,
            to_sq,
            promotion=promotion_map[promotion],
        )

        if move not in board.legal_moves:
            return illegal("Move is not legal in this position.")

        is_castle = data.get("IsCastle", False)

        if not isinstance(is_castle, bool):
            raise ValueError("IsCastle must be a boolean.")

        castle_side = data.get("CastleSide")
        actual_castle = board.is_castling(move)

        if is_castle != actual_castle:
            return illegal("IsCastle does not match the actual move.")

        if actual_castle:
            actual_side = (
                "King"
                if chess.square_file(to_sq) > chess.square_file(from_sq)
                else "Queen"
            )

            if castle_side != actual_side:
                return illegal("CastleSide does not match the actual move.")

        is_book_move = data.get("IsBookMove", False)

        if not isinstance(is_book_move, bool):
            raise ValueError("IsBookMove must be a boolean.")

        elo = data.get("Elo", 1500)

        if (
            isinstance(elo, bool)
            or not isinstance(elo, int)
            or not 100 <= elo <= 3500
        ):
            raise ValueError("Elo must be an integer between 100 and 3500.")

        mover = board.turn
        board_before = board.copy()

        with _engine_lock:
            engine = get_engine()
            strength = configure_strength(engine, elo)

            legal_moves_before = list(board.legal_moves)
            forced = len(legal_moves_before) == 1

            # Evaluation of the position before the submitted move.
            best_info = engine.analyse(
                board,
                chess.engine.Limit(time=ANALYSIS_TIME),
            )

            best_move = best_info.get("pv", [None])[0]
            best_score = score_cp(best_info, mover)

            captured_piece = board.piece_at(to_sq)

            board.push(move)

            kind = move_type(board_before, board, move)

            # Terminal positions must be scored from the mover's POV.
            if board.is_checkmate():
                played_score = 100000
                after_info = None

            elif (
                board.is_stalemate()
                or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()
            ):
                played_score = 0
                after_info = None

            else:
                after_info = engine.analyse(
                    board,
                    chess.engine.Limit(time=ANALYSIS_TIME),
                )

                played_score = score_cp(after_info, mover)

            cpl = max(0, best_score - played_score)

            # End-of-game ratings take priority.
            if board.is_checkmate():
                rating = (
                    "checkmate_white"
                    if mover == chess.WHITE
                    else "checkmate_black"
                )

            elif board.is_game_over():
                rating = (
                    "draw_white"
                    if mover == chess.WHITE
                    else "draw_black"
                )

            elif has_forced_mate(after_info, mover):
                rating = "mate"

            elif is_book_move:
                rating = "book"

            else:
                missed_win = (
                    best_score >= 300
                    and played_score <= 100
                    and cpl >= 150
                )

                brilliant = is_sacrifice(
                    board,
                    move,
                    after_info,
                    cpl,
                )

                great_find = (
                    move == best_move
                    and cpl <= 20
                    and captured_piece is not None
                    and captured_piece.piece_type in (
                        chess.ROOK,
                        chess.QUEEN,
                    )
                )

                rating = classify_rating(
                    cpl=cpl,
                    is_top_move=(move == best_move),
                    forced=forced,
                    missed_win=missed_win,
                    brilliant=brilliant,
                    great_find=great_find,
                )

            # The engine's reply.
            if board.is_game_over():
                bot_move_data = None

            else:
                reply_info = engine.analyse(
                    board,
                    chess.engine.Limit(time=ANALYSIS_TIME),
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
                            "White"
                            if bot_piece.color
                            else "Black"
                        ),
                        "Rating": "best",
                        "Promotion": (
                            chess.piece_name(reply.promotion).capitalize()
                            if reply.promotion
                            else None
                        ),
                    }

        return jsonify({
            "Legal": True,
            "Type": kind,
            "MoveRating": rating,
            "CentipawnLoss": cpl,
            "BotMove": bot_move_data,
            "EngineStrength": strength,
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
        port=int(os.getenv("PORT", "10000")),
    )
