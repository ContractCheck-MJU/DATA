"""테스트에서 `api.app`을 어느 위치에서 실행하든 import할 수 있게 한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
