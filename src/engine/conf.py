from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class COLUMNS:
    ADJ_CLOSE: Final = 'Adj Close'
    OPEN: Final = 'Open'
    HIGH: Final = 'High'
    LOW: Final = 'Low'
    CLOSE: Final = 'Close'
    VOLUME: Final = 'Volume'
