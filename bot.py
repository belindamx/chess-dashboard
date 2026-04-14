"""
BelBot — a style-simulated chess bot modelled on belindafails' historical games.
"""

from __future__ import annotations

import base64
import os
import random
import shutil
from typing import Optional

import chess
import chess.engine
import chess.svg

# ── Bot mode configs ─────────────────────────────────────────────────────────
BOT_MODES = {
    "Blitz Me":  {"skill": 10, "depth": 8,  "noise": 0.30, "label": "~1800–1900"},
    "Rapid Me":  {"skill": 13, "depth": 12, "noise": 0.18, "label": "~1900–2000"},
    "Best Me":   {"skill": 16, "depth": 15, "noise": 0.08, "label": "~2000–2100"},
}

# Board colours — teal dark squares to match the dashboard
BOARD_COLORS = {
    "square light":          "#e8f4f8",
    "square dark":           "#2ec4b6",
    "square light lastmove": "#b8e8e4",
    "square dark lastmove":  "#1a9e92",
    "margin":                "#1a2e35",
    "coord":                 "#c8f0ec",
}

# ECO prefix heuristics
_ECO_FIRST_WHITE = {"B": "e4", "C": "e4", "A": "d4", "D": "d4", "E": "d4"}
_E4_BLACK        = {"B": "c5", "C": "e5", "A": "Nf6", "D": "d6", "E": "Nf6"}
_D4_BLACK        = {"D": "d5", "E": "Nf6", "A": "Nf6", "B": "Nf6", "C": "Nf6"}

_PIECE_NAMES = {
    chess.PAWN:   "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK:   "rook",
    chess.QUEEN:  "queen",
    chess.KING:   "king",
}


# ── Stockfish discovery ───────────────────────────────────────────────────────
def find_stockfish() -> Optional[str]:
    """Return Stockfish binary path or None."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.isfile(env):
        return env
    which = shutil.which("stockfish")
    if which:
        return which
    candidates = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/opt/homebrew/Cellar/stockfish/16/bin/stockfish",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ── Style profile ─────────────────────────────────────────────────────────────
class StyleProfile:
    """Derived style tendencies from historical game data."""

    def __init__(self, df):
        import re
        import numpy as np

        white = df[df["my_color"] == "white"] if "my_color" in df.columns else df.iloc[:0]
        black = df[df["my_color"] == "black"] if "my_color" in df.columns else df.iloc[:0]

        # ECO prefix frequencies
        def _eco_prefs(sub):
            if "eco" not in sub.columns:
                return {}
            eco_raw = sub["eco"].dropna().astype(str)
            # strip URL, keep first letter
            prefix = eco_raw.str.replace(r"https?://[^/]+/openings/", "", regex=True).str[0]
            prefix = prefix[prefix.str.match(r"[A-E]")]
            if prefix.empty:
                return {}
            vc = prefix.value_counts(normalize=True)
            return vc.to_dict()

        self.white_eco_prefs = _eco_prefs(white)
        self.black_eco_prefs = _eco_prefs(black)

        # First move as white
        w_prefs = self.white_eco_prefs
        e4_weight = w_prefs.get("B", 0) + w_prefs.get("C", 0)
        d4_weight = w_prefs.get("A", 0) + w_prefs.get("D", 0) + w_prefs.get("E", 0)
        self.first_move_white = "e4" if e4_weight >= d4_weight else "d4"

        # Black responses
        b_prefs = self.black_eco_prefs
        top_b = max(b_prefs, key=b_prefs.get) if b_prefs else "B"
        self.black_resp_e4 = _E4_BLACK.get(top_b, "e5")
        self.black_resp_d4 = _D4_BLACK.get(top_b, "Nf6")

        # Accuracy
        if "my_accuracy" in df.columns:
            self.avg_accuracy = float(df["my_accuracy"].dropna().mean() or 75.0)
        else:
            self.avg_accuracy = 75.0

        # Rating per time control
        def _mean_rating(tc):
            if "time_class" not in df.columns or "my_rating" not in df.columns:
                return None
            sub = df[df["time_class"] == tc]["my_rating"].dropna()
            return int(sub.mean()) if len(sub) else None

        self.rating_bullet = _mean_rating("bullet") or 2000
        self.rating_blitz  = _mean_rating("blitz")  or 1850
        self.rating_rapid  = _mean_rating("rapid")  or 1750

        # Win rate
        self.win_rate = float(df["is_win"].mean()) if "is_win" in df.columns else 0.50

        # Top openings
        bad = {"Unknown", "Undefined", ""}
        def _top_openings(sub, n=3):
            if "opening_label" not in sub.columns:
                return []
            vc = (
                sub["opening_label"]
                .dropna()
                .pipe(lambda s: s[~s.isin(bad)])
                .pipe(lambda s: s[~s.str.match(r"^[A-E]\d{2}$", na=False)])
                .value_counts()
            )
            return vc.index[:n].tolist()

        self.top_white_openings = _top_openings(white)
        self.top_black_openings = _top_openings(black)


# ── BelBot ────────────────────────────────────────────────────────────────────
class BelBot:
    def __init__(self, profile: StyleProfile, mode: str = "Blitz Me"):
        self.profile  = profile
        self.mode     = mode
        self.cfg      = BOT_MODES.get(mode, BOT_MODES["Blitz Me"])
        self._sf_path = find_stockfish()

    # ── Public API ────────────────────────────────────────────────────────
    def get_move(self, board: chess.Board) -> tuple[Optional[chess.Move], str]:
        """Return (move, explanation). Move is None only if no legal moves."""
        if board.is_game_over():
            return None, ""

        # Opening book (first 2 full moves only)
        if board.fullmove_number <= 2:
            move, expl = self._opening_move(board)
            if move:
                return move, expl

        # Engine or fallback
        if self._sf_path:
            move, expl = self._engine_move(board)
        else:
            move, expl = self._fallback_move(board)

        return move, expl

    # ── Opening phase ─────────────────────────────────────────────────────
    def _opening_move(self, board: chess.Board):
        try:
            is_white_turn = board.turn == chess.WHITE
            fmn = board.fullmove_number

            # Move 1 as White
            if fmn == 1 and is_white_turn:
                mv = self._try_san(board, self.profile.first_move_white)
                if mv:
                    return mv, f"Opening with {self.profile.first_move_white} — a typical Belinda favourite."

            # Move 1 as Black (responding to white's move)
            if fmn == 1 and not is_white_turn:
                try:
                    last = board.peek()
                    to_file = chess.square_file(last.to_square)
                    to_rank = chess.square_rank(last.to_square)
                    if to_file == 4 and to_rank == 3:   # e4
                        resp = self.profile.black_resp_e4
                    elif to_file == 3 and to_rank == 3:  # d4
                        resp = self.profile.black_resp_d4
                    else:
                        resp = "Nf6"
                except Exception:
                    resp = "Nf6"
                mv = self._try_san(board, resp)
                if mv:
                    return mv, f"Responding with {resp} — consistent with Belinda's black repertoire."

        except Exception:
            pass
        return None, ""

    def _try_san(self, board: chess.Board, san: str) -> Optional[chess.Move]:
        try:
            mv = board.parse_san(san)
            if mv in board.legal_moves:
                return mv
        except Exception:
            pass
        return None

    # ── Engine move ───────────────────────────────────────────────────────
    def _engine_move(self, board: chess.Board):
        noise   = self.cfg["noise"]
        depth   = self.cfg["depth"]
        skill   = self.cfg["skill"]
        n_legal = board.legal_moves.count()
        if n_legal == 0:
            return None, ""
        multi   = min(3, n_legal)

        try:
            with chess.engine.SimpleEngine.popen_uci(self._sf_path) as sf:
                sf.configure({"Skill Level": skill})
                result = sf.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multi,
                )
                candidates = []
                for info in result:
                    pv = info.get("pv")
                    if pv:
                        candidates.append(pv[0])

            if not candidates:
                return self._fallback_move(board)

            # Weight: top move gets most probability; lower candidates absorb noise
            if len(candidates) == 1:
                move = candidates[0]
            elif len(candidates) == 2:
                weights = [1 - noise, noise]
                move = random.choices(candidates, weights=weights, k=1)[0]
            else:
                weights = [1 - noise, noise * 0.65, noise * 0.35]
                move = random.choices(candidates, weights=weights, k=1)[0]

            return move, self._explain(board, move)

        except Exception:
            return self._fallback_move(board)

    # ── Fallback (no Stockfish) ───────────────────────────────────────────
    def _fallback_move(self, board: chess.Board):
        legal = list(board.legal_moves)
        if not legal:
            return None, ""

        # Piece values
        VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        is_white = board.turn == chess.WHITE

        scores = []
        for mv in legal:
            score = 0.0
            piece = board.piece_at(mv.from_square)
            attacker_val = VAL.get(piece.piece_type, 0) if piece else 0

            # MVV-LVA: prefer capturing high-value pieces with low-value pieces
            if board.is_capture(mv):
                victim = board.piece_at(mv.to_square)
                if victim:
                    score += VAL.get(victim.piece_type, 0) * 10 - attacker_val

            board.push(mv)
            # Large penalty if the piece we just moved is now attacked (hanging)
            if board.is_attacked_by(board.turn, mv.to_square):
                score -= attacker_val * 8
            # Bonus for giving check
            if board.is_check():
                score += 5
            board.pop()

            # Development bonus (knights/bishops off back rank)
            if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                rank = chess.square_rank(mv.from_square)
                if (is_white and rank <= 1) or (not is_white and rank >= 6):
                    score += 3

            # Centre control bonus
            to_file = chess.square_file(mv.to_square)
            to_rank = chess.square_rank(mv.to_square)
            if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                score += 1

            # Modest noise so it doesn't play identically every game
            score += random.gauss(0, 3)
            scores.append(score)

        best = legal[scores.index(max(scores))]
        return best, self._explain(board, best)

    # ── Move explanation ──────────────────────────────────────────────────
    def _explain(self, board: chess.Board, move: chess.Move) -> str:
        if board.is_castling(move):
            return "Castling for king safety — a typical Belinda habit."
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            name = _PIECE_NAMES.get(victim.piece_type, "piece") if victim else "piece"
            return f"Capturing the {name}."
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            return "Giving check — keeping the pressure on."
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and board.fullmove_number <= 10:
            return "Developing a piece — Belinda likes an active opening."
        if piece and piece.piece_type == chess.PAWN:
            return "Pawn push — controlling the centre."
        return "Improving the position."


# ── Board rendering ───────────────────────────────────────────────────────────
def render_board(
    board: chess.Board,
    lastmove: Optional[chess.Move] = None,
    perspective: bool = chess.WHITE,
    size: int = 420,
) -> str:
    svg = chess.svg.board(
        board,
        lastmove=lastmove,
        orientation=perspective,
        size=size,
        colors=BOARD_COLORS,
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'width="{size}" height="{size}" '
        f'style="display:block;border-radius:4px;max-width:100%;" />'
    )


# ── Move history formatter ────────────────────────────────────────────────────
def fmt_move_history(history: list) -> str:
    """
    history: list of (san_string, is_white_move)
    Returns scrollable HTML move list in dashboard style.
    """
    if not history:
        return (
            '<div class="pb-history" style="color:#9ab8c0;">'
            'No moves yet.</div>'
        )

    # Pair moves: [(white_san, black_san_or_empty), ...]
    pairs = []
    buf   = []
    for san, is_white in history:
        if is_white:
            buf = [san, ""]
        else:
            if buf:
                buf[1] = san
            else:
                buf = ["...", san]
            pairs.append(tuple(buf))
            buf = []
    if buf and buf[0] != "":
        pairs.append((buf[0], ""))

    rows = []
    for i, (w, b) in enumerate(pairs, 1):
        rows.append(
            f'<span style="color:#2ec4b6;min-width:20px;display:inline-block;">{i}.</span>'
            f'<span style="color:#1a2e35;min-width:60px;display:inline-block;">{w}</span>'
            f'<span style="color:#1a2e35;">{b}</span>'
        )

    inner = "<br>".join(rows)
    return f'<div class="pb-history">{inner}</div>'
