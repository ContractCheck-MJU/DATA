"""Compatibility entry point: identical approved split as RoBERTa."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.train_tfidf_reviewed import main

if __name__ == '__main__':
    main()
