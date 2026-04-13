import os
import glob

def clean_duplicates():
    files = glob.glob("pickles_sleep_stage/**/*.pkl", recursive=True)
    groups = {}
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace('.pkl', '').split('_')
        if len(parts) < 4:
            continue
        
        # SSS19_SR_R_2730_5 -> parts = ['SSS19', 'SR', 'R', '2730', '5']
        # The stage is usually parts[-3], duration is parts[-2]
        stage = parts[-3]
        duration = int(parts[-2])
        bare_patient_id = "_".join(parts[:-3])
        
        # canonical patient id:
        if '-' in bare_patient_id:
            canon_id = bare_patient_id.split('-')[-1]
        else:
            canon_id = bare_patient_id
            
        key = (canon_id, stage)
        groups.setdefault(key, []).append((duration, f, bare_patient_id))
        
    for key, items in groups.items():
        if len(items) > 1:
            items.sort(key=lambda x: x[0], reverse=True)
            print(f"Group {key}:")
            print(f"  Keep: {items[0][1]} (duration={items[0][0]})")
            for item in items[1:]:
                print(f"  Remove: {item[1]} (duration={item[0]})")

clean_duplicates()
