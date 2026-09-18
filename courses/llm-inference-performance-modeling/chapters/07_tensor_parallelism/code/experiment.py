"""Run with torchrun: two-rank all-reduce calibration and held-out prediction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'shared'))
from communication import run


if __name__ == '__main__':
    run('07','all_reduce')
