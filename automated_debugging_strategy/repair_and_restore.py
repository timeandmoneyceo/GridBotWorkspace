import argparse
import os
import sys
import shutil
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BACKUPS_DIR = os.path.join(WORKSPACE_ROOT, 'backups')
QUARANTINE_DIR = os.path.join(BACKUPS_DIR, 'quarantine')

GOOD_MARKERS = [
    'class WebSocketManager',
    'class LSTMPricePredictor',
    'def compute_macd',
    'def compute_bollinger',
    'def fetch_trade_counts',
    'class BotCLI',
]

@dataclass
class Candidate:
    path: str
    size: int
    timestamp: Optional[datetime]
    score: float


def parse_backup_timestamp(name: str) -> Optional[datetime]:
    # Example: GridbotBackup.py.backup.20250920_200544
    m = re.search(r"\.backup\.(\d{8})_(\d{6})$", name)
    if not m:
        return None
    date_s, time_s = m[1], m[2]
    try:
        return datetime.strptime(date_s + time_s, '%Y%m%d%H%M%S')
    except Exception:
        return None


def list_backups_for(target_filename: str) -> List[str]:
    if not os.path.isdir(BACKUPS_DIR):
        return []
    base = os.path.basename(target_filename)
    pattern = f'{base}.backup.'
    return [os.path.join(BACKUPS_DIR, f) for f in os.listdir(BACKUPS_DIR) if f.startswith(pattern)]


def has_good_markers(path: str) -> int:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return sum(bool(m in content)
               for m in GOOD_MARKERS)
    except Exception:
        return 0


def rank_backups(backups: List[str]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for p in backups:
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        ts = parse_backup_timestamp(os.path.basename(p))
        marker_hits = has_good_markers(p)
        # heuristic score: markers weighted + size factor + recency
        recency = (ts.timestamp() if ts else 0) / 1e10
        score = marker_hits * 2.0 + (size / 100000.0) + recency
        candidates.append(Candidate(path=p, size=size, timestamp=ts, score=score))
    # sort by score desc then by timestamp desc
    candidates.sort(key=lambda c: (c.score, c.timestamp or datetime.min), reverse=True)
    return candidates


def restore_from_backup(target_path: str, backup_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    # quarantine current
    if os.path.exists(target_path):
        q_name = f'{os.path.basename(target_path)}.quarantine.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        q_path = os.path.join(QUARANTINE_DIR, q_name)
        if not dry_run:
            shutil.copy2(target_path, q_path)
        print(f"Quarantined current file to: {q_path}")
    # restore
    if not dry_run:
        shutil.copy2(backup_path, target_path)
    print(f"Restored {target_path} from backup {backup_path}")
    return True, backup_path


def sanity_compare(target_path: str, backup_path: str) -> None:
    try:
        cur_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        bk_size = os.path.getsize(backup_path)
        print(f"Current size: {cur_size} bytes | Backup size: {bk_size} bytes")
        hits_cur = has_good_markers(target_path) if os.path.exists(target_path) else 0
        hits_bk = has_good_markers(backup_path)
        print(f"Markers: current={hits_cur}/{len(GOOD_MARKERS)} | backup={hits_bk}/{len(GOOD_MARKERS)}")
    except Exception as e:
        print(f"Sanity compare failed: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Repair and Restore from backups')
    parser.add_argument('--target', required=True, help='Absolute or workspace-relative path to the target file')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without writing')
    parser.add_argument('--print-backups', action='store_true', help='Only list ranked backups for inspection')
    args = parser.parse_args(argv)

    # normalize target path
    target = args.target
    if not os.path.isabs(target):
        # interpret relative to workspace root
        target = os.path.join(WORKSPACE_ROOT, target)
    target = os.path.normpath(target)

    backups = list_backups_for(target)
    if not backups:
        print(f"No backups found for {target} in {BACKUPS_DIR}")
        return 2

    ranked = rank_backups(backups)
    print("Ranked backups (top 5):")
    for c in ranked[:5]:
        ts = c.timestamp.strftime('%Y-%m-%d %H:%M:%S') if c.timestamp else 'unknown'
        print(f"- score={c.score:.2f} | size={c.size} | ts={ts} | {c.path}")

    if args.print_backups:
        return 0

    best = ranked[0] if ranked else None
    if not best:
        print("No suitable backup candidates.")
        return 3

    print("\nSanity before restore:")
    sanity_compare(target, best.path)

    ok, used = restore_from_backup(target, best.path, dry_run=args.dry_run)
    if not ok:
        print("Restore failed")
        return 4

    if not args.dry_run:
        print("\nSanity after restore:")
        sanity_compare(target, used)

    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
