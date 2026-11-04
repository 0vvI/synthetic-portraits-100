"""Single-writer harvest: copy (never move) the newest Gemini download into portraits/<set>/<id>.jpeg and log it.
Usage: HARVEST_DIRS=~/Downloads python3 harvest.py identities ID001 [note] [epoch_before_send]
Refuses if the newest download is older than 20 minutes or already harvested (sha256 seen)."""
import sys, glob, os, shutil, hashlib, time, json
setname, ident, chat = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
after = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # epoch seconds recorded just before the item was sent
DIRS = os.environ.get("HARVEST_DIRS", "~/Downloads").split(":")   # where the browser saves Gemini downloads
dl = sorted((p for d in DIRS for ext in ("jpeg","jpg","png") for p in glob.glob(os.path.expanduser(f"{d}/Gemini_Generated_Image_*.{ext}"))), key=os.path.getmtime)
assert dl, "no Gemini download found"
src = dl[-1]
age = time.time() - os.path.getmtime(src)
assert age < 1200, f"newest download is {age/60:.1f} min old; refusing (stale)"
assert os.path.getmtime(src) >= after, f"newest download ({os.path.basename(src)}) predates this item's send time; refusing (orphan from an earlier item)"
h = hashlib.sha256(open(src, "rb").read()).hexdigest()
logp = "harvest_log.jsonl"
seen = {json.loads(l)["sha256"] for l in open(logp)} if os.path.exists(logp) else set()
assert h not in seen, "newest download already harvested; nothing new landed"
dst = f"portraits/{setname}/{ident}.jpeg"
assert not os.path.exists(dst), f"{dst} exists; refusing to overwrite"
shutil.copy2(src, dst)
with open(logp, "a") as f:
    f.write(json.dumps({"id": ident, "set": setname, "src": os.path.basename(src), "sha256": h, "chat": chat, "bytes": os.path.getsize(dst)}) + "\n")
print("harvested", dst, os.path.getsize(dst), "bytes", h[:12])
