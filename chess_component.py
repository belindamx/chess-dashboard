"""Streamlit custom component — interactive drag-and-drop chess board."""
import streamlit.components.v1 as components
from pathlib import Path
from typing import Optional

_COMPONENT_PATH = Path(__file__).parent / "components" / "chess_board"

_chess_board_cmp = components.declare_component(
    "chess_board",
    path=str(_COMPONENT_PATH),
)

def chess_board(
    fen: str,
    orientation: str = "white",
    last_from: Optional[str] = None,
    last_to: Optional[str] = None,
    height: int = 460,
    key: str = None,
) -> Optional[str]:
    """
    Render an interactive chess board. Returns a UCI move string when the
    player makes a move (e.g. 'e2e4'), otherwise None.
    """
    return _chess_board_cmp(
        fen=fen,
        orientation=orientation,
        last_from=last_from,
        last_to=last_to,
        default=None,
        key=key,
    )
