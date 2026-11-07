from pydantic import BaseModel
from typing import Optional
import chess

class SquareData(BaseModel):
    Piece: str
    Color: str


class AnalyzeRequest(BaseModel):
    board: dict[str, SquareData]

    old: str
    new: str
    piece: str
    color: str
    elo: int

    castling: str = "-"
    en_passant: Optional[str] = None
    halfmove: int = 0
    fullmove: int = 1


PIECE_TYPES = {
    "Pawn": chess.PAWN,
    "Knight": chess.KNIGHT,
    "Bishop": chess.BISHOP,
    "Rook": chess.ROOK,
    "Queen": chess.QUEEN,
    "King": chess.KING,
}


def create_board(data: AnalyzeRequest):
    board = chess.Board.empty()

    for square_name, info in data.board.items():
        square_name = square_name.lower()

        if square_name not in chess.SQUARE_NAMES:
            raise ValueError(f"Invalid square: {square_name}")

        if info.Piece not in PIECE_TYPES:
            raise ValueError(f"Invalid piece: {info.Piece}")

        if info.Color not in ("White", "Black"):
            raise ValueError(f"Invalid color: {info.Color}")

        piece = chess.Piece(
            PIECE_TYPES[info.Piece],
            info.Color == "White"
        )

        board.set_piece_at(
            chess.parse_square(square_name),
            piece
        )

    board.turn = data.color == "White"

    board.set_castling_fen(data.castling)

    if data.en_passant:
        board.ep_square = chess.parse_square(
            data.en_passant.lower()
        )

    board.halfmove_clock = data.halfmove
    board.fullmove_number = data.fullmove

    return board
