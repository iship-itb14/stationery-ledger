"""Injects data/results.json into site/template.html -> site/index.html (static, deploy anywhere)."""
import json
d=json.dumps(json.load(open("data/results.json")),separators=(",",":")).replace("</","<\\/")
open("site/index.html","w").write(open("site/template.html").read().replace("__DATA__",d)); print("built site/index.html")
