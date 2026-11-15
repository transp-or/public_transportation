"""Loopback-only HTTP API for on-demand gravity viewer measurement queries."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

from public_transportation.inference.temporal_assignment_persistence import (
    open_temporal_block_artifact,
)

from .estimator import GravityEstimationResult
from .measurement_od_contributions import MeasurementODContributionQuery
from .reporting import GRAVITY_REPORT_PROVENANCE_FIELDS
from .viewer_bundle import (
    GravityViewerBundle,
    _validate_linear_mapping_for_bundle,
    read_gravity_viewer_bundle,
)


class GravityViewerQueryService:
    """Validated query source shared by the package-local HTTP handler."""

    def __init__(
        self,
        *,
        bundle: GravityViewerBundle,
        fit_manifest_path: str | Path,
        mapping_artifact_path: str | Path,
        rho: float | None = None,
    ) -> None:
        self.bundle = bundle
        fit_path = Path(fit_manifest_path).expanduser().resolve()
        try:
            fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"fit manifest is missing or corrupt: {fit_path}") from error
        if not isinstance(fit_payload, dict) or not isinstance(fit_payload.get("result"), dict):
            raise ValueError("fit manifest must contain a result object.")
        fit_result = GravityEstimationResult.from_dict(fit_payload["result"])
        bundle_manifest = bundle.manifest
        if fit_result.model_fingerprint != bundle_manifest.get("model_fingerprint"):
            raise ValueError("fit manifest model fingerprint differs from the selected viewer bundle.")
        bundle_provenance = bundle_manifest.get("provenance", {})
        if not isinstance(bundle_provenance, dict):
            raise ValueError("viewer bundle provenance is invalid.")
        for field in GRAVITY_REPORT_PROVENANCE_FIELDS:
            if fit_payload.get(field) != bundle_provenance.get(field):
                raise ValueError(f"fit manifest {field} differs from the selected viewer bundle.")

        summary = bundle.viewer_summary.get("measurement_od_contributions", {})
        if (
            not isinstance(summary, dict)
            or not summary.get("available")
            or summary.get("query_mode") != "local_service"
        ):
            raise ValueError("viewer bundle is not configured for on-demand measurement queries.")

        artifact = open_temporal_block_artifact(mapping_artifact_path)
        if summary.get("mapping_manifest_sha256") != artifact.manifest_sha256:
            raise ValueError("mapping manifest checksum differs from the selected viewer bundle.")
        for field, actual in (
            ("fit_fingerprint", fit_result.model_fingerprint),
            ("assignment_fingerprint", artifact.identity.timetable_fingerprint),
            ("mapping_artifact_fingerprint", artifact.identity.fingerprint),
            (
                "canonical_index_fingerprint",
                artifact.canonical_index.artifact_fingerprint,
            ),
            ("binding_fingerprint", artifact.canonical_index.binding_fingerprint),
            (
                "od_layout_fingerprint",
                artifact.canonical_index.source_od_layout_fingerprint,
            ),
        ):
            bundle_expected = (
                bundle_manifest.get("model_fingerprint")
                if field == "fit_fingerprint"
                else bundle_manifest.get("artifact_identity_fingerprint")
                if field == "mapping_artifact_fingerprint"
                else bundle_provenance.get(field)
            )
            if summary.get(field) != actual or bundle_expected != actual:
                raise ValueError(f"mapping {field} differs from the selected fit and viewer bundle.")

        report_path = bundle.path / "report.json"
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("viewer bundle report.json is missing or corrupt.") from error
        if not isinstance(report_payload, dict):
            raise ValueError("viewer bundle report.json is invalid.")
        _validate_linear_mapping_for_bundle(
            artifact=artifact,
            fit_result=fit_result,
            report_payload=report_payload,
            metadata={},
            full_od_path=bundle.path / "full_od.csv",
        )
        self.query = MeasurementODContributionQuery(
            artifact=artifact,
            fit_result=fit_result,
            rho=rho,
            max_cached_pages=16,
            max_page_size=500,
        )
        if self.query.cache_namespace != (
            str(summary.get("fit_fingerprint")),
            str(summary.get("mapping_artifact_fingerprint")),
        ):
            raise ValueError("on-demand query cache identity differs from the viewer bundle.")

    def query_measurement(self, measurement_index: int, *, page: int, page_size: int) -> dict[str, object]:
        return self.query.query(
            measurement_index,
            page=page,
            page_size=page_size,
        )

    def query_od(self, od_index: int, *, page: int, page_size: int) -> dict[str, object]:
        return self.query.query_od(od_index, page=page, page_size=page_size)


def _handler_for(service: GravityViewerQueryService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GravityViewerQuery/1"

        def _json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            origin = self.headers.get("Origin")
            if origin == "null" or (origin and _is_loopback_origin(origin)):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            origin = self.headers.get("Origin")
            if origin != "null" and not (origin and _is_loopback_origin(origin)):
                self._json(403, {"error": "Only local viewer origins are allowed."})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._json(
                    200,
                    {
                        "service": "gravity_viewer_measurement_queries",
                        "fit_fingerprint": service.query.cache_namespace[0],
                        "mapping_artifact_fingerprint": service.query.cache_namespace[1],
                    },
                )
                return
            if parsed.path not in {
                "/api/measurement-od-contributions",
                "/api/od-measurement-contributions",
            }:
                self._json(404, {"error": "Unknown local gravity viewer query endpoint."})
                return
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                page = _single_int(query, "page", default=0)
                page_size = _single_int(query, "page_size", default=100)
                if parsed.path == "/api/measurement-od-contributions":
                    measurement_index = _single_int(query, "measurement_index")
                    payload = service.query_measurement(measurement_index, page=page, page_size=page_size)
                else:
                    od_index = _single_int(query, "od_index")
                    payload = service.query_od(od_index, page=page, page_size=page_size)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                self._json(400, {"error": str(error)})
                return
            self._json(200, payload)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep local service output to one concise request line per query.
            super().log_message(format, *args)

    return Handler


def _single_int(query: dict[str, list[str]], name: str, *, default: int | None = None) -> int:
    values = query.get(name)
    if values is None:
        if default is None:
            raise KeyError(f"missing query parameter {name!r}.")
        return default
    if len(values) != 1:
        raise ValueError(f"query parameter {name!r} must be specified once.")
    try:
        return int(values[0])
    except ValueError as error:
        raise ValueError(f"query parameter {name!r} must be an integer.") from error


def _is_loopback_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve bounded on-demand OD contributions for a local gravity viewer bundle."
    )
    parser.add_argument("--bundle", required=True, type=Path, help="viewer bundle directory")
    parser.add_argument("--fit-manifest", required=True, type=Path, help="persisted fit manifest")
    parser.add_argument(
        "--mapping-artifact",
        required=True,
        type=Path,
        help="persisted temporal mapping artifact directory",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="exact fitted rho only when observation scales are absent",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    bundle = read_gravity_viewer_bundle(args.bundle)
    service = GravityViewerQueryService(
        bundle=bundle,
        fit_manifest_path=args.fit_manifest,
        mapping_artifact_path=args.mapping_artifact,
        rho=args.rho,
    )
    server = HTTPServer(("127.0.0.1", args.port), _handler_for(service))
    print(f"Gravity viewer query API listening at http://127.0.0.1:{args.port}")
    print("Open tools/gravity_viewer/index.html, select the bundle folder, and use Measurement mode.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local gravity viewer query API.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
