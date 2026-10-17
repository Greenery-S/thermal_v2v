"""Download the TartanRGBT modalities needed for thermal_v2v and extract them in place.

Default subset: rgb_in_thermal (RGB warped into the thermal-left frame), thermal_left_rect_16 (16-bit target),
thermal_left_rect_8 (carries the FFC/NUC shutter flags, thermal_left_ffc/), target_timestamps + calibration/configs.
The full repo is ~1.0 TB (stereo_depth, ZED, right thermal); we need ~300 GB extracted for the 38 sequences.

Usage: python scripts/data/download_tartanrgbt.py --out <dir> --sequences_file configs/data/tartanrgbt_download.txt \
           [--delete_zips]
"""
import argparse, os, zipfile, sys, time
from huggingface_hub import snapshot_download

DEFAULT = ["rgb_in_thermal", "thermal_left_rect_16", "thermal_left_rect_8"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modalities", nargs="+", default=DEFAULT)
    ap.add_argument("--delete_zips", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sequences_file", default=None, help="optional list of <day>/<folder> to restrict to")
    a = ap.parse_args()
    pats = ["README.md", "calibration/*", "configs/*", "*.csv", "data_extraction.py"]
    if not a.sequences_file:
        pats.append("*/*/target_timestamps.txt")
    if a.sequences_file:
        seqs = [l.split("#")[0].strip() for l in open(a.sequences_file)]
        seqs = [x for x in seqs if x]
        # skip zips already extracted (stamp file): with --delete_zips a re-run would otherwise download them again
        done = lambda q, m: os.path.exists(os.path.join(a.out, q, f"{m}.zip.extracted"))
        pats += [f"{q}/target_timestamps.txt" for q in seqs] + [f"{q}/{m}.zip" for q in seqs for m in a.modalities if not done(q, m)]
    else:
        pats += [f"*/*/{m}.zip" for m in a.modalities]
    for attempt in range(20):
        try:
            snapshot_download("theairlabcmu/TartanRGBT", repo_type="dataset", local_dir=a.out,
                              allow_patterns=pats, max_workers=a.workers)
            break
        except Exception as e:  # transient HF errors: retry, already-downloaded files are skipped
            print(f"[retry {attempt}] {e!r}", flush=True); time.sleep(60)
    else:
        sys.exit("download failed after retries")
    zips = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.out) for f in fs if f.endswith(".zip"))
    for z in zips:
        seq = os.path.dirname(z)
        stamp = z + ".extracted"
        if os.path.exists(stamp):
            continue
        print("extract", z, flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(seq)
        open(stamp, "w").close()
        if a.delete_zips:
            os.remove(z)
    print("DONE", len(zips), "zips", flush=True)

if __name__ == "__main__":
    main()
