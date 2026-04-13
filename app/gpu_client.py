"""Client for communicating with the GPU cluster 3D reconstruction service.

Handles multi-view job submission, polling, preview fetching, and
GLB download.  The GPU cluster exposes a FastAPI service (see
gpu-cluster/api/main.py) at the URL configured by GPU_CLUSTER_URL.

New API flow (3-step job submission):
    1. POST /jobs              → create a job
    2. POST /jobs/{id}/upload/{view}  → upload each view image
    3. POST /jobs/{id}/start   → enqueue for processing

3-view canonical setup:
    - front:  perpendicular, centered
    - side:   perpendicular from the right
    - top:    bird's-eye looking straight down
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx

from .config import config

logger = logging.getLogger("brickedup.gpu_client")

# Canonical view names — must match gpu-cluster/api/models.py ViewLabel
CANONICAL_VIEWS = ["front", "side", "top"]

# Polling configuration
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 600  # 10 minutes max

# Map GPU cluster job statuses to approximate progress percentages
_STATUS_PROGRESS: dict[str, int] = {
    "pending": 0,
    "preprocessing": 10,
    "camera_init": 20,
    "coarse_recon": 40,
    "subject_isolation": 60,
    "trellis_completion": 75,
    "exporting": 90,
    "completed": 100,
    "failed": 0,
}


class GPUClusterError(Exception):
    """Raised when the GPU cluster returns an error or is unreachable."""


# ──────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────


async def check_gpu_health() -> dict:
    """Check GPU cluster connectivity and health.

    Returns:
        Dict with keys: status, url, detail.
    """
    url = config.GPU_CLUSTER_URL.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
            data = resp.json()
            raw_status = data.get("status", "unknown")
            # Normalise: the GPU cluster returns "ok" when healthy
            status = "healthy" if raw_status == "ok" else raw_status
            return {
                "status": status,
                "url": url,
                "detail": "",
            }
    except Exception as e:
        return {
            "status": "unreachable",
            "url": url,
            "detail": str(e),
        }


# ──────────────────────────────────────────────────────────────────────
# Job submission (3-step: create → upload views → start)
# ──────────────────────────────────────────────────────────────────────


async def submit_multiview_job(
    generated_views: dict[str, str],
    category: str = "generic_object",
    pipeline: str = "canonical_mv_hybrid",
    params: Optional[dict] = None,
) -> str:
    """Submit a multi-view reconstruction job to the GPU cluster.

    Uses the new 3-step API:
      1. POST /jobs           – create a job
      2. POST /jobs/{id}/upload/{view} – upload each view image
      3. POST /jobs/{id}/start – enqueue for processing

    Args:
        generated_views: Dict mapping view name → local file path.
            Expected keys: front, side, top.
        category: Object category for reconstruction hints (unused in new API).
        pipeline: GPU cluster pipeline to use (unused in new API).
        params: Optional pipeline config overrides.

    Returns:
        Job ID from the GPU cluster.

    Raises:
        GPUClusterError: If submission fails.
    """
    url = config.GPU_CLUSTER_URL.rstrip("/")

    # Validate that all views exist locally
    view_files: dict[str, Path] = {}
    for view_name in CANONICAL_VIEWS:
        path = generated_views.get(view_name)
        if not path:
            raise GPUClusterError(
                f"Missing required view '{view_name}' in generated_views"
            )
        filepath = Path(path)
        if not filepath.exists():
            raise GPUClusterError(
                f"View file not found: {path}"
            )
        view_files[view_name] = filepath

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Create job
            create_body: dict = {}
            if params:
                # Pass params as pipeline config overrides
                create_body["config"] = params

            resp = await client.post(f"{url}/jobs", json=create_body)
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data.get("job_id")
            if not job_id:
                raise GPUClusterError(
                    f"GPU cluster did not return a job_id: {job_data}"
                )
            logger.info("Created GPU cluster job %s", job_id)

            # Step 2: Upload each view
            for view_name, filepath in view_files.items():
                with open(filepath, "rb") as fh:
                    files = {"file": (filepath.name, fh, "image/png")}
                    resp = await client.post(
                        f"{url}/jobs/{job_id}/upload/{view_name}",
                        files=files,
                    )
                    resp.raise_for_status()
                    logger.info(
                        "Uploaded %s view for job %s (%s)",
                        view_name, job_id, filepath.name,
                    )

            # Step 3: Start processing
            resp = await client.post(f"{url}/jobs/{job_id}/start")
            resp.raise_for_status()
            logger.info("Started GPU cluster job %s", job_id)

            return job_id

    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        raise GPUClusterError(
            f"GPU cluster returned {e.response.status_code}: {detail}"
        ) from e
    except httpx.RequestError as e:
        raise GPUClusterError(
            f"Could not connect to GPU cluster at {url}: {e}"
        ) from e


# ──────────────────────────────────────────────────────────────────────
# Polling
# ──────────────────────────────────────────────────────────────────────


async def poll_gpu_job(
    job_id: str,
    on_progress: Optional[callable] = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    timeout: float = POLL_TIMEOUT_SECONDS,
) -> dict:
    """Poll a GPU cluster job until completion or failure.

    Args:
        job_id: Job ID to poll.
        on_progress: Optional callback(status_str, progress_int).
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait.

    Returns:
        Final job status dict.

    Raises:
        GPUClusterError: If the job fails or times out.
    """
    url = config.GPU_CLUSTER_URL.rstrip("/")
    elapsed = 0.0

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < timeout:
            try:
                resp = await client.get(f"{url}/jobs/{job_id}/status")
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning("Poll error for job %s: %s", job_id, e)
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                continue

            status = data.get("status", "unknown")
            progress = _STATUS_PROGRESS.get(status, 0)

            if on_progress:
                on_progress(status, progress)

            if status == "completed":
                logger.info("GPU job %s completed", job_id)
                return data

            if status == "failed":
                error = data.get("error_message", "Unknown error")
                raise GPUClusterError(
                    f"GPU job {job_id} failed: {error}"
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    raise GPUClusterError(
        f"GPU job {job_id} timed out after {timeout}s"
    )


# ──────────────────────────────────────────────────────────────────────
# Preview fetching
# ──────────────────────────────────────────────────────────────────────


async def get_preprocessing_previews(job_id: str) -> dict[str, str]:
    """Fetch preview image URLs from the GPU cluster.

    Lists artifacts and builds URLs for preview-related images
    (preprocessed views, masks, etc.).

    Returns:
        Dict mapping preview name → full URL for download.
    """
    url = config.GPU_CLUSTER_URL.rstrip("/")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{url}/jobs/{job_id}/artifacts")
        resp.raise_for_status()
        data = resp.json()

    previews: dict[str, str] = {}
    artifacts = data.get("artifacts", [])

    for artifact_path in artifacts:
        # Build preview entries for preprocessed images and masks
        parts = Path(artifact_path).parts
        artifact_url = f"{url}/jobs/{job_id}/artifacts/{artifact_path}"

        if len(parts) >= 2:
            stage = parts[0]  # e.g. "preprocessed", "isolation"
            filename = Path(artifact_path).stem  # e.g. "front"

            if stage == "preprocessed" and _is_image(artifact_path):
                key = f"preprocessed_{filename}"
                previews[key] = artifact_url
            elif stage == "isolation" and "masks" in parts and _is_image(artifact_path):
                key = f"mask_{filename}"
                previews[key] = artifact_url
            elif stage == "isolation" and "masked_images" in parts and _is_image(artifact_path):
                key = f"masked_{filename}"
                previews[key] = artifact_url

    return previews


def _is_image(path: str) -> bool:
    """Check if a path looks like an image file."""
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ──────────────────────────────────────────────────────────────────────
# GLB download
# ──────────────────────────────────────────────────────────────────────


async def download_glb(job_id: str, output_path: str) -> None:
    """Download the final GLB output from the GPU cluster.

    Searches the job's artifacts for a .glb file and downloads it.
    Falls back to known paths (trellis/trellis_output.glb, export/*.glb).

    Args:
        job_id: Completed job ID.
        output_path: Local path to save the GLB file.

    Raises:
        GPUClusterError: If download fails.
    """
    url = config.GPU_CLUSTER_URL.rstrip("/")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # First, find the GLB artifact path
            glb_artifact = await _find_glb_artifact(client, url, job_id)
            if not glb_artifact:
                raise GPUClusterError(
                    f"No GLB artifact found for job {job_id}"
                )

            # Download the GLB file
            resp = await client.get(
                f"{url}/jobs/{job_id}/artifacts/{glb_artifact}"
            )
            resp.raise_for_status()
            out.write_bytes(resp.content)

        logger.info("Downloaded GLB for job %s → %s", job_id, output_path)
    except GPUClusterError:
        raise
    except httpx.HTTPStatusError as e:
        raise GPUClusterError(
            f"Failed to download GLB: HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        raise GPUClusterError(
            f"Failed to download GLB from {url}: {e}"
        ) from e


async def _find_glb_artifact(
    client: httpx.AsyncClient,
    base_url: str,
    job_id: str,
) -> Optional[str]:
    """Find the GLB artifact path within a job's storage.

    Checks known paths first, then falls back to listing all artifacts.
    """
    # Try known paths first (faster than listing all artifacts)
    known_paths = [
        "trellis/trellis_output.glb",
        "export/model.glb",
        "export/output.glb",
    ]

    for path in known_paths:
        try:
            resp = await client.head(
                f"{base_url}/jobs/{job_id}/artifacts/{path}"
            )
            if resp.status_code == 200:
                return path
        except Exception:
            continue

    # Fall back to listing artifacts and finding any .glb file
    try:
        resp = await client.get(f"{base_url}/jobs/{job_id}/artifacts")
        resp.raise_for_status()
        data = resp.json()
        artifacts = data.get("artifacts", [])

        for artifact_path in artifacts:
            if artifact_path.endswith(".glb"):
                return artifact_path
    except Exception as e:
        logger.warning("Failed to list artifacts for job %s: %s", job_id, e)

    return None

