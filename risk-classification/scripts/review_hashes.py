"""Print text fingerprints for human confirmation; never edit decisions or labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data_contract import read_rows, row_digest
from scripts.export_approved import DECISIONS

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--decisions',type=Path,default=DECISIONS)
    a=p.parse_args()
    for row in read_rows(a.decisions):
        if row.get('reviewed_text','').strip():
            print(row['record_id'], row_digest({'text':row['reviewed_text']}))
