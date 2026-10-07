import argparse
import json
import sys

from . import pipeline
from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="agenttool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    a = sub.add_parser("analyze")
    a.add_argument("--script", required=True, help="path to script .txt")
    a.add_argument("--audio", required=True)
    a.add_argument("--project", required=True)
    a.add_argument("--settings", help="settings JSON file")
    r = sub.add_parser("render")
    r.add_argument("--project", required=True)
    r.add_argument("--settings")
    r.add_argument("--out")
    args = ap.parse_args()
    settings = json.load(open(args.settings)) if getattr(args, "settings", None) else {}
    if args.cmd == "serve":
        serve()
    elif args.cmd == "analyze":
        plan = pipeline.analyze(open(args.script).read(), args.audio, args.project, settings)
        print(f"{len(plan['scenes'])} scenes -> {args.project}/plan.json", file=sys.stderr)
    else:
        print(pipeline.render_plan(args.project, settings, args.out))


main()
