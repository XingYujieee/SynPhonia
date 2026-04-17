"""CLI helpers for provider management."""

from __future__ import annotations

import argparse
import json
import sys

from .provider_registry import get_registry


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m lite_synphonia providers",
        description="Manage API providers for LiteSynphonia.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List configured providers.")

    add = sub.add_parser("add", help="Add or update a provider.")
    add.add_argument("name", type=str, help="Provider name, e.g. deepgram.")
    add.add_argument("--base-url", required=True, type=str, help="Provider base URL.")
    add.add_argument("--model", default="", type=str, help="Default model ID.")
    add.add_argument("--api-key", required=True, type=str, help="API key/token.")
    add.add_argument(
        "--service",
        action="append",
        default=[],
        help="Service tag (repeatable), e.g. transcription/summarization/embedding.",
    )
    add.add_argument("--timeout-seconds", type=float, default=60.0)
    add.add_argument("--max-retries", type=int, default=3)
    add.add_argument("--temperature", type=float, default=0.1)

    rm = sub.add_parser("remove", help="Remove a provider by name.")
    rm.add_argument("name", type=str)

    show = sub.add_parser("show", help="Show provider details.")
    show.add_argument("name", type=str)
    show.add_argument("--raw", action="store_true", help="Show full API key.")

    return p


def run_providers_command(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    registry = get_registry()

    if args.command == "list":
        providers = registry.list()
        if not providers:
            print("[lite_synphonia] No providers configured.")
            return 0
        print(f"[lite_synphonia] Providers ({len(providers)}):")
        for p in providers:
            svc = ",".join(p.services) if p.services else "-"
            key_hint = p.public_dict().get("api_key", "")
            print(
                f"  - {p.name:12s}  base={p.base_url:30s} "
                f"model={p.model_id or '-':18s} services={svc:20s} key={key_hint}"
            )
        return 0

    if args.command == "add":
        entry = registry.upsert(
            name=args.name,
            base_url=args.base_url,
            model_id=args.model,
            api_key=args.api_key,
            services=args.service,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
        print(
            "[lite_synphonia] Provider saved: "
            f"name={entry.name} base={entry.base_url} model={entry.model_id or '-'}"
        )
        return 0

    if args.command == "remove":
        ok = registry.remove(args.name)
        if ok:
            print(f"[lite_synphonia] Removed provider: {args.name}")
            return 0
        print(f"[lite_synphonia] Provider not found: {args.name}")
        return 1

    if args.command == "show":
        try:
            entry = registry.get(args.name)
        except KeyError as exc:
            print(f"[lite_synphonia] {exc}", file=sys.stderr)
            return 1
        payload = entry.to_dict() if args.raw else entry.public_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1
