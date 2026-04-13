import os, glob
from collections import defaultdict
import sys

base_dir = sys.argv[1]
if os.path.exists(base_dir):
    for stage in os.listdir(base_dir):
        stage_dir = os.path.join(base_dir, stage)
        if not os.path.isdir(stage_dir): continue
        files = glob.glob(os.path.join(stage_dir, "*.pkl"))
        groups = defaultdict(list)
        for f in files:
            bn = os.path.basename(f)
            parts = bn.split("_")
            if len(parts) >= 3:
                groups[parts[0]].append(f)
        for pid, flist in groups.items():
            if len(flist) > 1:
                flist.sort(key=lambda x: int(os.path.basename(x).split("_")[2]) if os.path.basename(x).split("_")[2].isdigit() else 0, reverse=True)
                for fdel in flist[1:]:
                    print(f"Would remove: {fdel}")
