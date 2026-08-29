"""
Storage abstraction used by every pipeline. Three backends, auto-selected
by which credentials are present in the environment (checked in this
order):

  1. Hopsworks       -> if HOPSWORKS_API_KEY is set
  2. Backblaze B2     -> if B2_KEY_ID / B2_APPLICATION_KEY / B2_BUCKET_NAME
                          / B2_ENDPOINT_URL are all set
  3. Local disk        -> fallback, no credentials needed

Every object this module writes to B2 is placed under a "b2-store/"
prefix, so the storage backend that produced a file is visible directly
in its path/name inside the bucket (e.g. "b2-store/models/aqi_model_24h/v3/model.joblib").

IMPORTANT for model loading: when a cloud backend (Hopsworks or B2) is
configured, load_model_and_columns() always checks the cloud store FIRST
for the newest model version, before ever touching a local cache. This
matters for a deployed Streamlit app: the app process stays alive across
many page loads, so if it read a locally-cached joblib file first, it
would keep serving yesterday's model forever and never notice that
training_pipeline.py retrained and pushed a new version. Local disk is
only used as a last-resort fallback if the cloud call itself fails.
"""
import os
from dotenv import load_dotenv
import io
import json
import pathlib
import joblib
import pandas as pd

LOCAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "local_store"
B2_PREFIX = "b2-store"
# load_dotenv()
# A very common cause of "it silently falls back to local even though I set
# the B2 vars" is that they live in a .env file that never actually gets
# loaded into the process -> os.environ.get() sees nothing. Auto-load one
# if present, searching from this file's location up to the repo root.






load_dotenv()
print(f"[storage] loaded environment variables")



B2_REQUIRED_VARS = ["B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME", "B2_ENDPOINT_URL"]


def hopsworks_available():
    return bool(os.environ.get("HOPSWORKS_API_KEY"))


def b2_available():
    return all(os.environ.get(v) for v in B2_REQUIRED_VARS)


def _diagnose_b2_env():
    """Prints exactly which B2 vars are missing, so a typo or an
    unloaded .env file is obvious instead of a silent local-storage
    fallback."""
    missing = [v for v in B2_REQUIRED_VARS if not os.environ.get(v)]
    present = [v for v in B2_REQUIRED_VARS if os.environ.get(v)]
    if missing and present:
        print(f"[storage] B2 is PARTIALLY configured -> falling back to local storage. "
              f"Present: {present}. Missing: {missing}. "
              f"Check for typos in variable names, or that your .env file is actually "
              f"being loaded (see the '[storage] loaded environment variables from...' "
              f"message above -> if you don't see that line, your .env isn't being picked up).")


_BACKEND_ANNOUNCED = False


def get_backend():
    global _BACKEND_ANNOUNCED
    if hopsworks_available():
        backend = "hopsworks"
    elif b2_available():
        backend = "b2"
    else:
        _diagnose_b2_env()
        backend = "local"

    if not _BACKEND_ANNOUNCED:
        print(f"[storage] using backend: '{backend}'")
        _BACKEND_ANNOUNCED = True
    return backend


def _b2_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=120,
            max_pool_connections=10,
        ),
    )


def _retryable_exceptions():
    """Imported lazily so this module doesn't hard-require botocore/boto3
    when running in local-only mode."""
    import ssl
    import botocore.exceptions as bce
    return (
        ssl.SSLEOFError,
        ssl.SSLError,
        ConnectionError,
        OSError,
        bce.SSLError,                 # what actually surfaces once botocore's
        bce.EndpointConnectionError,   # own internal retries are exhausted
        bce.ConnectionClosedError,
    )


def _with_retries(fn, *args, max_attempts=6, **kwargs):
    """Backstop retry for errors that slip past (or survive) botocore's own
    retry handling. A large multipart upload getting an SSLEOFError on one
    part still raises botocore.exceptions.SSLError after botocore's own
    retries for that part are exhausted -> this wraps the whole call one
    level up and retries the operation again from here."""
    import time
    retryable = _retryable_exceptions()
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except retryable as e:
            last_exc = e
            wait = min(2 ** attempt, 30)
            print(f"[storage] transient error on attempt {attempt}/{max_attempts} "
                  f"({e!r}); retrying in {wait}s...")
            time.sleep(wait)
    raise last_exc


def _b2_bucket():
    return os.environ["B2_BUCKET_NAME"]


def _b2_list_versions(client, bucket, prefix):
    """Returns sorted list of integer version numbers found under
    f"{prefix}/v<N>/" style keys."""
    resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
    versions = set()
    for obj in resp.get("Contents", []):
        # key looks like: b2-store/models/aqi_model_24h/v3/model.joblib
        parts = obj["Key"][len(prefix) + 1:].split("/")
        if parts and parts[0].startswith("v") and parts[0][1:].isdigit():
            versions.add(int(parts[0][1:]))
    return sorted(versions)


class FeatureStore:
    def __init__(self):
        self.backend = get_backend()
        if self.backend == "hopsworks":
            import hopsworks
            self.project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
            self.fs = self.project.get_feature_store()
        elif self.backend == "b2":
            self.client = _b2_client()
            self.bucket = _b2_bucket()
        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    def save_features(self, df, name, version=1, primary_key=None):
        primary_key = primary_key or ["city", "timestamp"]

        if self.backend == "hopsworks":
            fg = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                event_time="timestamp",
                online_enabled=False,
                description=f"Auto-generated AQI features ({name})",
            )
            fg.insert(df, write_options={"wait_for_job": False})

        elif self.backend == "b2":
            key = f"{B2_PREFIX}/features/{name}_v{version}.parquet"
            try:
                obj = _with_retries(self.client.get_object, Bucket=self.bucket, Key=key)
                old = pd.read_parquet(io.BytesIO(obj["Body"].read()))
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=primary_key, keep="last")
            except self.client.exceptions.NoSuchKey:
                pass
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            _with_retries(self.client.put_object, Bucket=self.bucket, Key=key, Body=buf.getvalue())

        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            path = LOCAL_DIR / f"{name}_v{version}.parquet"
            if path.exists():
                old = pd.read_parquet(path)
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=primary_key, keep="last")
            df.to_parquet(path, index=False)

    def load_features(self, name, version=1):
        if self.backend == "hopsworks":
            fg = self.fs.get_feature_group(name=name, version=version)
            return fg.read()

        elif self.backend == "b2":
            key = f"{B2_PREFIX}/features/{name}_v{version}.parquet"
            try:
                obj = _with_retries(self.client.get_object, Bucket=self.bucket, Key=key)
                return pd.read_parquet(io.BytesIO(obj["Body"].read()))
            except self.client.exceptions.NoSuchKey:
                return pd.DataFrame()

        else:
            path = LOCAL_DIR / f"{name}_v{version}.parquet"
            if not path.exists():
                return pd.DataFrame()
            return pd.read_parquet(path)


class ModelRegistry:
    def __init__(self):
        self.backend = get_backend()
        if self.backend == "hopsworks":
            import hopsworks
            self.project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
            self.mr = self.project.get_model_registry()
        elif self.backend == "b2":
            self.client = _b2_client()
            self.bucket = _b2_bucket()
        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, name, metrics=None, feature_cols=None):
        if self.backend == "b2":
            self._save_model_b2(model, name, metrics, feature_cols)
        elif self.backend == "hopsworks":
            self._save_model_hopsworks(model, name, metrics, feature_cols)
        else:
            self._save_model_local(model, name, metrics, feature_cols)

    def _save_model_b2(self, model, name, metrics, feature_cols):
        """Entirely in-memory -> nothing ever touches local disk. Models
        are small now (a few MB after hyperparameter tuning + joblib
        compression), so a single put_object per file is simpler and
        just as reliable as the old multipart upload_file() approach,
        without needing any local staging file at all."""
        prefix = f"{B2_PREFIX}/models/{name}"
        existing_versions = _b2_list_versions(self.client, self.bucket, prefix)
        next_version = (max(existing_versions) + 1) if existing_versions else 1
        version_prefix = f"{prefix}/v{next_version}"

        model_buf = io.BytesIO()
        joblib.dump(model, model_buf, compress=3)
        _with_retries(self.client.put_object, Bucket=self.bucket,
                      Key=f"{version_prefix}/model.joblib", Body=model_buf.getvalue())

        if feature_cols is not None:
            _with_retries(self.client.put_object, Bucket=self.bucket,
                          Key=f"{version_prefix}/columns.json",
                          Body=json.dumps(feature_cols).encode())
        if metrics is not None:
            _with_retries(self.client.put_object, Bucket=self.bucket,
                          Key=f"{version_prefix}/metrics.json",
                          Body=json.dumps(metrics, indent=2).encode())

        print(f"[storage] uploaded '{name}' v{next_version} to B2 "
              f"({model_buf.tell() / 1e6:.2f}MB) -- no local copy kept")

    def _save_model_hopsworks(self, model, name, metrics, feature_cols):
        """Hopsworks' SDK requires an actual local path to upload from ->
        use a temp directory that's automatically deleted the moment the
        `with` block exits, so nothing lingers on disk afterward."""
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
            tmp_dir = pathlib.Path(tmp)
            joblib.dump(model, tmp_dir / "model.joblib", compress=3)
            if feature_cols is not None:
                with open(tmp_dir / "columns.json", "w") as f:
                    json.dump(feature_cols, f)
            if metrics is not None:
                with open(tmp_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

            hw_model = self.mr.python.create_model(
                name=name,
                metrics=metrics or {},
                description=f"Auto-trained model {name}",
            )
            hw_model.save(str(tmp_dir))
        print(f"[storage] uploaded '{name}' to Hopsworks (temp staging deleted)")

    def _save_model_local(self, model, name, metrics, feature_cols):
        """Only used when NO cloud backend is configured -> this is the
        actual persistent store in that mode, so it's meant to stick
        around (and is what GitHub Actions commits back to the repo)."""
        bundle_dir = LOCAL_DIR / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, bundle_dir / "model.joblib", compress=3)
        if feature_cols is not None:
            with open(bundle_dir / "columns.json", "w") as f:
                json.dump(feature_cols, f)
        if metrics is not None:
            with open(bundle_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        print(f"[storage] saved '{name}' locally to {bundle_dir}")

    def _download_latest_from_hopsworks(self, name):
        """Downloads to a Hopsworks-managed temp dir, loads into memory,
        then deletes that temp dir immediately -> nothing persists
        locally afterward, matching the same policy as B2."""
        import shutil
        models = self.mr.get_models(name)
        if not models:
            raise LookupError(f"No versions of model '{name}' found in Hopsworks yet.")
        latest = max(models, key=lambda m: m.version)
        model_dir = pathlib.Path(latest.download())
        try:
            model = joblib.load(model_dir / "model.joblib")
            feature_cols = None
            cols_path = model_dir / "columns.json"
            if cols_path.exists():
                with open(cols_path) as f:
                    feature_cols = json.load(f)
            return model, feature_cols
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)

    def _download_latest_from_b2(self, name):
        """Pure in-memory download via BytesIO -> never writes to disk
        at all, so there's nothing to delete afterward."""
        prefix = f"{B2_PREFIX}/models/{name}"
        versions = _b2_list_versions(self.client, self.bucket, prefix)
        if not versions:
            raise LookupError(f"No versions of model '{name}' found in bucket yet.")
        latest_version = max(versions)
        version_prefix = f"{prefix}/v{latest_version}"

        model_obj = _with_retries(
            self.client.get_object, Bucket=self.bucket, Key=f"{version_prefix}/model.joblib"
        )
        model = joblib.load(io.BytesIO(model_obj["Body"].read()))

        feature_cols = None
        try:
            cols_obj = _with_retries(
                self.client.get_object, Bucket=self.bucket, Key=f"{version_prefix}/columns.json"
            )
            feature_cols = json.loads(cols_obj["Body"].read())
        except self.client.exceptions.NoSuchKey:
            pass

        return model, feature_cols

    def load_model_and_columns(self, name):
        """Preferred accessor: returns (model, feature_cols) fetched
        together so they can never mismatch, or (None, None) if no
        model has been trained yet for this name -> that's a normal,
        expected state (not every horizon may be trained yet), so it's
        handled quietly here rather than raising.

        NOTE on behavior change: when a cloud backend is configured,
        this NO LONGER falls back to a stale local cache if the cloud
        call fails for any OTHER reason (auth, connectivity, etc.) ->
        per your requirement that models should live only in the cloud
        store (never locally or in git) when one is configured, a
        genuinely failed cloud call now raises instead of silently
        serving old data. The retry logic in _with_retries already
        absorbs transient network blips, so an exception here means a
        real, persistent problem worth seeing rather than hiding."""
        if self.backend == "hopsworks":
            try:
                return self._download_latest_from_hopsworks(name)
            except LookupError:
                return None, None
        elif self.backend == "b2":
            try:
                return self._download_latest_from_b2(name)
            except LookupError:
                return None, None

        bundle_dir = LOCAL_DIR / name
        model_path = bundle_dir / "model.joblib"
        cols_path = bundle_dir / "columns.json"
        if not model_path.exists():
            return None, None
        model = joblib.load(model_path)
        feature_cols = None
        if cols_path.exists():
            with open(cols_path) as f:
                feature_cols = json.load(f)
        return model, feature_cols

    def load_model(self, name):
        """Kept for convenience; prefer load_model_and_columns()."""
        model, _ = self.load_model_and_columns(name)
        return model

    def load_feature_columns(self, name):
        """Kept for convenience; prefer load_model_and_columns()."""
        _, feature_cols = self.load_model_and_columns(name)
        return feature_cols


REPORTS_PREFIX = "reports"


class ReportStore:
    """Small artifacts that aren't models or feature rows: SHAP plot
    images, training summary/comparison JSON, and per-pipeline run-status
    logs. Same policy as everything else here: if a cloud backend (B2)
    is configured, these go there and nowhere else (not local disk, not
    git). If no cloud backend is configured, they live under
    local_store/reports/, which IS what GitHub Actions commits back to
    the repo in that fallback mode.

    Hopsworks note: this intentionally falls back to local storage for
    reports even when Hopsworks is your model/feature backend, since a
    generic small-file store isn't part of the interfaces already used
    elsewhere in this project. If you want these in Hopsworks too, its
    dataset API (project.get_dataset_api()) can be wired in here later.
    """

    def __init__(self):
        self.backend = get_backend() if b2_available() else "local"
        if self.backend == "b2":
            self.client = _b2_client()
            self.bucket = _b2_bucket()
        else:
            (LOCAL_DIR / REPORTS_PREFIX).mkdir(parents=True, exist_ok=True)

    def save_bytes(self, name, data: bytes):
        if self.backend == "b2":
            key = f"{B2_PREFIX}/{REPORTS_PREFIX}/{name}"
            _with_retries(self.client.put_object, Bucket=self.bucket, Key=key, Body=data)
        else:
            path = LOCAL_DIR / REPORTS_PREFIX / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def save_json(self, name, obj):
        self.save_bytes(name, json.dumps(obj, indent=2).encode())

    def load_bytes(self, name):
        if self.backend == "b2":
            key = f"{B2_PREFIX}/{REPORTS_PREFIX}/{name}"
            try:
                obj = _with_retries(self.client.get_object, Bucket=self.bucket, Key=key)
                return obj["Body"].read()
            except self.client.exceptions.NoSuchKey:
                return None
        else:
            path = LOCAL_DIR / REPORTS_PREFIX / name
            if not path.exists():
                return None
            return path.read_bytes()

    def load_json(self, name):
        data = self.load_bytes(name)
        return json.loads(data) if data is not None else None

    def list_names(self, prefix=""):
        if self.backend == "b2":
            full_prefix = f"{B2_PREFIX}/{REPORTS_PREFIX}/{prefix}"
            resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
            base_len = len(f"{B2_PREFIX}/{REPORTS_PREFIX}/")
            return sorted(o["Key"][base_len:] for o in resp.get("Contents", []))
        else:
            base = LOCAL_DIR / REPORTS_PREFIX
            if not base.exists():
                return []
            return sorted(str(p.relative_to(base)) for p in base.glob(f"{prefix}*"))


def log_run_status(pipeline_name, status, details=None):
    """Called from each pipeline's __main__ block. Writes a small JSON
    status file (via ReportStore, so it ends up wherever your other
    reports live) that the dashboard's pipeline-status panel reads --
    this is the lightweight in-app substitute for tailing GitHub Actions
    logs directly. Full step-by-step logs always remain available
    natively on GitHub: your repo -> Actions tab -> pick a workflow run.
    """
    from datetime import datetime, timezone, timedelta
    entry = {
        "pipeline": pipeline_name,
        "status": status,
        "timestamp": datetime.now(timezone(timedelta(hours=5))).isoformat(),
        "details": details or {},
    }
    try:
        ReportStore().save_json(f"status_{pipeline_name}.json", entry)
    except Exception as e:
        # Never let logging itself break the pipeline run.
        print(f"[storage] could not write run-status log: {e}")
