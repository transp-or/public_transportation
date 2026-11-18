"""Single-process local gravity viewer and on-demand measurement query server."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

from tqdm import tqdm

from public_transportation.inference.temporal_assignment_persistence import (
    open_temporal_block_artifact_for_viewer,
)
from public_transportation.inference.hierarchical_artifacts import package_revision

from .estimator import GravityEstimationResult
from .measurement_od_contributions import (
    MeasurementODContributionQuery,
    MeasurementODQueryCancelled,
)
from .reporting import GRAVITY_REPORT_PROVENANCE_FIELDS
from .viewer_bundle import (
    GravityViewerBundle,
    _validate_linear_mapping_for_bundle,
    read_gravity_viewer_bundle,
)

_StartupProgressCallback = Callable[[str, int, int, str], None]


def _elapsed_label(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes}m {remainder}s" if minutes else f"{remainder}s"


@contextmanager
def _startup_stage(label: str, *, heartbeat_seconds: float = 10.0) -> Iterator[None]:
    """Print startup stages and elapsed-time heartbeats for long operations."""
    started = time.monotonic()
    stop = threading.Event()
    tqdm.write(f"[startup] {label}…", file=sys.stderr)

    def report_while_running() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = _elapsed_label(time.monotonic() - started)
            tqdm.write(
                f"[startup] Still {label} ({elapsed} elapsed).",
                file=sys.stderr,
            )

    reporter = threading.Thread(target=report_while_running, daemon=True)
    reporter.start()
    try:
        yield
    except BaseException:
        tqdm.write(
            f"[startup] Failed: {label} after {_elapsed_label(time.monotonic() - started)}.",
            file=sys.stderr,
        )
        raise
    else:
        tqdm.write(
            f"[startup] Done: {label} ({_elapsed_label(time.monotonic() - started)}).",
            file=sys.stderr,
        )
    finally:
        stop.set()
        reporter.join(timeout=1.0)


class _StartupProgressDisplay:
    """Render startup scan callbacks with tqdm bars and ETA estimates."""

    def __init__(self) -> None:
        self._bars: dict[str, tqdm] = {}

    def __call__(self, stage: str, completed: int, total: int, detail: str) -> None:
        if total <= 0:
            suffix = f" — {detail}" if detail else ""
            tqdm.write(f"[startup] {stage}{suffix}…", file=sys.stderr)
            return
        bar = self._bars.get(stage)
        if bar is None or bar.total != total:
            if bar is not None:
                bar.close()
            unit = "rows" if " rows" in stage else "records"
            unit_scale = False
            if "checksum" in stage.lower():
                unit = "B"
                unit_scale = True
            bar = tqdm(
                total=total,
                desc=f"[startup] {stage}",
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=1024,
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stderr,
                leave=True,
                disable=False,
            )
            self._bars[stage] = bar
        target = min(max(0, completed), total)
        if target > bar.n:
            bar.update(target - bar.n)
        if detail:
            bar.set_postfix_str(detail, refresh=False)
        if target >= total:
            bar.close()
            self._bars.pop(stage, None)


class GravityViewerQueryService:
    """Validated query source shared by the package-local HTTP handler."""

    def __init__(
        self,
        *,
        bundle: GravityViewerBundle,
        fit_manifest_path: str | Path,
        mapping_artifact_path: str | Path,
        rho: float | None = None,
        progress_callback: _StartupProgressCallback | None = None,
    ) -> None:
        self.bundle = bundle
        fit_path = Path(fit_manifest_path).expanduser().resolve()
        with _startup_stage("reading and decoding fit manifest"):
            try:
                fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"fit manifest is missing or corrupt: {fit_path}") from error
        if not isinstance(fit_payload, dict) or not isinstance(fit_payload.get("result"), dict):
            raise ValueError("fit manifest must contain a result object.")
        with _startup_stage("restoring fitted model state"):
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
        if summary.get("query_descriptor_version") != 2:
            raise ValueError("viewer bundle uses an unsupported on-demand query descriptor version.")
        runtime_revision = package_revision()
        if (
            summary.get("runtime_package_revision") != runtime_revision
            or bundle_manifest.get("runtime_package_revision") != runtime_revision
        ):
            raise ValueError("viewer bundle runtime package revision differs from the query-service runtime.")
        fit_source_revision = fit_payload.get("package_revision")
        if not isinstance(fit_source_revision, str) or not fit_source_revision:
            raise ValueError("fit manifest is missing its source package revision.")

        with _startup_stage("opening temporal mapping artifact and checking its files"):
            artifact = open_temporal_block_artifact_for_viewer(
                mapping_artifact_path,
                fit_source_package_revision=fit_source_revision,
                progress_callback=progress_callback,
            )
        self.artifact = artifact
        self.time_zone = bundle_manifest.get("time_zone")
        if summary.get("mapping_manifest_sha256") != artifact.manifest_sha256:
            raise ValueError("mapping manifest checksum differs from the selected viewer bundle.")
        for field, actual in (
            ("fit_fingerprint", fit_result.model_fingerprint),
            ("assignment_fingerprint", bundle_provenance.get("assignment_fingerprint")),
            ("mapping_fingerprint", bundle_provenance.get("mapping_fingerprint")),
            ("artifact_timetable_fingerprint", artifact.identity.timetable_fingerprint),
            (
                "artifact_measurement_mapping_fingerprint",
                artifact.identity.measurement_mapping_fingerprint,
            ),
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
            if summary.get(field) != actual:
                raise ValueError(f"mapping query {field} differs from the selected fit and viewer bundle.")
        if bundle_manifest.get("model_fingerprint") != summary.get("fit_fingerprint"):
            raise ValueError("viewer query fit fingerprint differs from bundle metadata.")
        if bundle_manifest.get("artifact_identity_fingerprint") != summary.get("mapping_artifact_fingerprint"):
            raise ValueError("viewer query full mapping-artifact identity differs from bundle metadata.")
        if summary.get("mapping_artifact_schema_version") != artifact.artifact_schema_version:
            raise ValueError("mapping artifact schema version differs from the viewer bundle.")
        if (
            summary.get("fit_source_package_revision") != fit_source_revision
            or summary.get("mapping_source_package_revision") != artifact.source_package_revision
            or artifact.source_package_revision != fit_source_revision
            or bundle_manifest.get("fit_source_package_revision") != fit_source_revision
            or bundle_manifest.get("mapping_source_package_revision") != artifact.source_package_revision
        ):
            raise ValueError("viewer fit and mapping source package revisions differ.")

        report_path = bundle.path / "report.json"
        with _startup_stage("reading viewer report metadata"):
            try:
                report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("viewer bundle report.json is missing or corrupt.") from error
        if not isinstance(report_payload, dict):
            raise ValueError("viewer bundle report.json is invalid.")
        with _startup_stage("checking fit, mapping, and bundle identities"):
            _validate_linear_mapping_for_bundle(
                artifact=artifact,
                fit_result=fit_result,
                report_payload=report_payload,
                metadata={},
                full_od_path=bundle.path / "full_od.csv",
                validate_block_payloads=False,
                progress_callback=progress_callback,
            )
        with _startup_stage("initializing on-demand contribution query"):
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
        # Keep the expensive sparse-mapping queries serial. The threaded HTTP
        # server can still serve the viewer assets and health endpoint while a
        # contribution query is running.
        self._query_lock = threading.Lock()
        self._query_progress_lock = threading.Lock()
        self._query_progress: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._query_request_lock = threading.Lock()
        self._query_requests: dict[str, threading.Event] = {}
        self._cancelled_query_ids: set[str] = set()

    def _register_query(self, request_id: str | None) -> tuple[str, threading.Event]:
        identifier = request_id or uuid.uuid4().hex
        if len(identifier) > 128:
            raise ValueError("request_id must be at most 128 characters.")
        event = threading.Event()
        with self._query_request_lock:
            if identifier in self._query_requests:
                raise ValueError("request_id is already active.")
            if identifier in self._cancelled_query_ids:
                event.set()
                self._cancelled_query_ids.discard(identifier)
            self._query_requests[identifier] = event
        return identifier, event

    def _finish_query(self, request_id: str) -> None:
        with self._query_request_lock:
            self._query_requests.pop(request_id, None)

    def cancel_query(self, request_id: str) -> bool:
        """Cancel an active query, or remember cancellation if it has not started yet."""
        if not request_id or len(request_id) > 128:
            raise ValueError("request_id must contain 1 to 128 characters.")
        with self._query_request_lock:
            event = self._query_requests.get(request_id)
            if event is not None:
                event.set()
            else:
                self._cancelled_query_ids.add(request_id)
                if len(self._cancelled_query_ids) > 256:
                    self._cancelled_query_ids.pop()
        return True

    def _set_query_progress(
        self,
        measurement_index: int,
        *,
        request_id: str,
        status: str,
        phase: str,
        completed: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        with self._query_progress_lock:
            self._query_progress[request_id] = {
                "request_id": request_id,
                "measurement_index": measurement_index,
                "status": status,
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
                "fit_fingerprint": self.query.cache_namespace[0],
                "mapping_artifact_fingerprint": self.query.cache_namespace[1],
            }
            self._query_progress.move_to_end(request_id)
            while len(self._query_progress) > 256:
                self._query_progress.popitem(last=False)

    def query_progress(self, measurement_index: int, request_id: str) -> dict[str, object]:
        """Return a lock-free snapshot of the active/most recent measurement query."""
        with self._query_progress_lock:
            progress = dict(self._query_progress.get(request_id, {}))
        if (
            progress.get("measurement_index") != measurement_index
            or progress.get("request_id") != request_id
        ):
            return {
                "measurement_index": measurement_index,
                "request_id": request_id,
                "status": "idle",
                "phase": "waiting",
                "completed": 0,
                "total": 0,
                "message": "Waiting for the selected measurement query",
                "fit_fingerprint": self.query.cache_namespace[0],
                "mapping_artifact_fingerprint": self.query.cache_namespace[1],
            }
        return progress

    def query_measurement(
        self,
        measurement_index: int,
        *,
        page: int,
        page_size: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        request_id, cancel_event = self._register_query(request_id)
        self._set_query_progress(
            measurement_index,
            request_id=request_id,
            status="waiting",
            phase="waiting",
            message="Waiting for the previous query to stop",
        )
        try:
            with self._query_lock:
                if cancel_event.is_set():
                    raise MeasurementODQueryCancelled("Query cancelled because the selection changed.")
                self._set_query_progress(
                    measurement_index,
                    request_id=request_id,
                    status="running",
                    phase="starting",
                    message="Starting the measurement-row query",
                )
                payload = self.query.query(
                    measurement_index,
                    page=page,
                    page_size=page_size,
                    cancel_check=cancel_event.is_set,
                    progress_callback=lambda phase, completed, total, message: self._set_query_progress(
                        measurement_index,
                        request_id=request_id,
                        status="running",
                        phase=phase,
                        completed=completed,
                        total=total,
                        message=message,
                    ),
                )
                if cancel_event.is_set():
                    raise MeasurementODQueryCancelled("Query cancelled because the selection changed.")
                self._set_query_progress(
                    measurement_index,
                    request_id=request_id,
                    status="running",
                    phase="validating",
                    message="Checking the query against the saved measurement prediction",
                )
                selected = self.bundle.read_table(
                    "predicted_measurements.csv",
                    columns=("row_index", "observed", "modeled"),
                    filters={"row_index": measurement_index},
                    chunksize=1,
                )
                if cancel_event.is_set():
                    raise MeasurementODQueryCancelled("Query cancelled because the selection changed.")
                if len(selected) != 1:
                    raise ValueError("selected measurement row is missing or duplicated in the viewer bundle.")
                observed = float(selected.iloc[0]["observed"])
                modeled_mean = float(selected.iloc[0]["modeled"])
                if not math.isfinite(observed) or not math.isfinite(modeled_mean):
                    raise ValueError("selected measurement observation or modeled mean is non-finite.")
                components = payload.get("components")
                if isinstance(components, dict) and not math.isclose(
                    float(components.get("modeled_mean", float("nan"))),
                    modeled_mean,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ):
                    raise ValueError("query decomposition modeled mean differs from the saved viewer prediction.")
                payload["observed"] = observed
                payload["modeled_mean"] = modeled_mean
                self._set_query_progress(
                    measurement_index,
                    request_id=request_id,
                    status="complete",
                    phase="complete",
                    completed=1,
                    total=1,
                    message="OD contributions are ready",
                )
                return payload
        except Exception as error:
            if cancel_event.is_set() and not isinstance(error, MeasurementODQueryCancelled):
                error = MeasurementODQueryCancelled("Query cancelled because the selection changed.")
            if isinstance(error, MeasurementODQueryCancelled):
                self._set_query_progress(
                    measurement_index,
                    request_id=request_id,
                    status="cancelled",
                    phase="cancelled",
                    message=str(error),
                )
                raise error
            self._set_query_progress(
                measurement_index,
                request_id=request_id,
                status="failed",
                phase="failed",
                message=str(error),
            )
            raise
        finally:
            self._finish_query(request_id)

    def query_od(
        self,
        od_index: int,
        *,
        page: int,
        page_size: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        request_id, cancel_event = self._register_query(request_id)
        try:
            with self._query_lock:
                if cancel_event.is_set():
                    raise MeasurementODQueryCancelled("Query cancelled because the selection changed.")
                return self.query.query_od(
                    od_index,
                    page=page,
                    page_size=page_size,
                    cancel_check=cancel_event.is_set,
                )
        except Exception as error:
            if cancel_event.is_set() and not isinstance(error, MeasurementODQueryCancelled):
                raise MeasurementODQueryCancelled("Query cancelled because the selection changed.") from error
            raise
        finally:
            self._finish_query(request_id)


def _handler_for(
    service: GravityViewerQueryService,
    viewer_directory: Path,
) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        server_version = "GravityViewer/1"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(viewer_directory), **kwargs)

        def end_headers(self) -> None:
            if not urlparse(self.path).path.startswith("/api/"):
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

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
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The browser may abort a superseded query before its handler returns.
                pass

        def do_OPTIONS(self) -> None:  # noqa: N802
            origin = self.headers.get("Origin")
            if origin != "null" and not (origin and _is_loopback_origin(origin)):
                self._json(403, {"error": "Only local viewer origins are allowed."})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin or "null")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                super().do_GET()
                return
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
            if parsed.path == "/api/departure-intervals":
                self._json(
                    200,
                    {
                        "intervals": [
                            {
                                "interval_id": interval.interval_id,
                                "start_seconds": interval.start_seconds,
                                "end_seconds": interval.end_seconds,
                            }
                            for interval in service.artifact.canonical_index.time_intervals
                        ],
                        "time_zone": service.time_zone,
                        "fit_fingerprint": service.query.cache_namespace[0],
                        "mapping_artifact_fingerprint": service.query.cache_namespace[1],
                    },
                )
                return
            if parsed.path == "/api/measurement-query-progress":
                try:
                    query = parse_qs(parsed.query, strict_parsing=True)
                    measurement_index = _single_int(query, "measurement_index")
                    request_id = _single_string(query, "request_id")
                    self._json(200, service.query_progress(measurement_index, request_id))
                except (KeyError, TypeError, ValueError) as error:
                    self._json(400, {"error": str(error)})
                return
            if parsed.path not in {
                "/api/measurement-od-contributions",
                "/api/od-measurement-contributions",
            }:
                self._json(404, {"error": "Unknown local gravity viewer query endpoint."})
                return
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                request_id = _single_string(query, "request_id")
                page = _single_int(query, "page", default=0)
                page_size = _single_int(query, "page_size", default=100)
                if parsed.path == "/api/measurement-od-contributions":
                    measurement_index = _single_int(query, "measurement_index")
                    payload = service.query_measurement(
                        measurement_index,
                        page=page,
                        page_size=page_size,
                        request_id=request_id,
                    )
                else:
                    od_index = _single_int(query, "od_index")
                    payload = service.query_od(od_index, page=page, page_size=page_size, request_id=request_id)
            except MeasurementODQueryCancelled as error:
                self._json(409, {"error": str(error), "cancelled": True})
                return
            except (KeyError, TypeError, ValueError, IndexError) as error:
                self._json(400, {"error": str(error)})
                return
            self._json(200, payload)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/cancel-query":
                self._json(404, {"error": "Unknown local gravity viewer query endpoint."})
                return
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                request_id = _single_string(query, "request_id")
                service.cancel_query(request_id)
            except (KeyError, TypeError, ValueError) as error:
                self._json(400, {"error": str(error)})
                return
            self._json(200, {"request_id": request_id, "cancelled": True})

        def log_message(self, format: str, *args: Any) -> None:
            # Keep local server output to one concise request line per request.
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


def _single_string(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if values is None:
        raise KeyError(f"missing query parameter {name!r}.")
    if len(values) != 1 or not values[0]:
        raise ValueError(f"query parameter {name!r} must be specified once and non-empty.")
    return values[0]


def _is_loopback_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _default_viewer_directory() -> Path:
    """Find the static viewer assets in a public_transportation checkout."""
    repository_root = Path(__file__).resolve().parents[4]
    return repository_root / "tools" / "gravity_viewer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the local gravity viewer and its bounded on-demand OD query API "
            "from one loopback-only process."
        )
    )
    parser.add_argument(
        "--viewer-directory",
        type=Path,
        default=None,
        help="static viewer asset directory (defaults to tools/gravity_viewer in this checkout)",
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
    viewer_directory = (
        args.viewer_directory.expanduser().resolve()
        if args.viewer_directory is not None
        else _default_viewer_directory()
    )
    if not viewer_directory.is_dir() or not (viewer_directory / "index.html").is_file():
        parser.error(
            f"viewer assets were not found in {viewer_directory}; "
            "pass --viewer-directory pointing to tools/gravity_viewer."
        )
    print("Preparing the local gravity viewer service.", file=sys.stderr, flush=True)
    progress_display = _StartupProgressDisplay()
    with _startup_stage("loading and validating viewer bundle"):
        bundle = read_gravity_viewer_bundle(
            args.bundle,
            progress_callback=progress_display,
        )
    service = GravityViewerQueryService(
        bundle=bundle,
        fit_manifest_path=args.fit_manifest,
        mapping_artifact_path=args.mapping_artifact,
        rho=args.rho,
        progress_callback=progress_display,
    )
    with _startup_stage("binding local web server"):
        server = ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            _handler_for(service, viewer_directory),
        )
    print(f"Gravity viewer and query API listening at http://127.0.0.1:{args.port}/")
    print("Select the bundle folder in the viewer and use Measurement mode.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local gravity viewer query API.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
