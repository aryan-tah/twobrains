"""Bundle web/index.html + selected runs into one self-contained HTML (for sharing / claude.ai artifact).
usage: uv run python experiments/build_artifact.py out.html run1 [run2 ...]"""
import sys, json, re
out, runs = sys.argv[1], sys.argv[2:]
html = open("web/index.html").read()
data = {r: json.load(open(f"web/data/{r}.json")) for r in runs}
for d in data.values():                      # the page never reads per-neuron type strings; drop them to fit the size budget
    for br in d["brains"].values(): br.pop("type", None)
inject = f"<script>window.TWOBRAINS_RUNS={json.dumps(runs)};window.TWOBRAINS_DATA={json.dumps(data, separators=(',', ':'))};</script>\n"
html = html.replace("<script>\nconst RUNS", inject + "<script>\nconst RUNS")
# artifact host wraps the page itself: strip doctype/html/head/body wrappers, keep title+style+content
html = re.sub(r"^<!doctype html>\s*<html[^>]*>\s*<head>\s*<meta[^>]*>\s*<meta[^>]*>\s*", "", html, flags=re.I)
html = html.replace("</head>\n<body>", "").replace("</body>\n</html>", "")
open(out, "w").write(html); print("wrote", out, round(len(html) / 1e6, 1), "MB")
