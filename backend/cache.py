"""
Prediction Cache for Satellite Scene Classification.

Provides a pluggable cache mechanism:
- AzureTablePredictionCache: Used in Azure Container Apps across all backend replicas.
- SQLitePredictionCache: Used for local development and offline persistence.
- InMemoryPredictionCache: Used for fast, isolated automated testing.

Cache keys are composed of:
  PartitionKey / Key: model_version
  RowKey / SubKey: sha256(raw_image_bytes)

Cached value:
  The averaged post-TTA softmax probability vector (list of floats).
  This allows dynamic thresholding without re-running model inference.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from azure.core.exceptions import (  # type: ignore[import-untyped]
        HttpResponseError,
        ResourceExistsError,
        ResourceNotFoundError,
    )
    from azure.data.tables import TableClient, UpdateMode  # type: ignore[import-untyped]
    from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    HttpResponseError = Exception  # type: ignore[misc, assignment]
    ResourceExistsError = Exception  # type: ignore[misc, assignment]
    ResourceNotFoundError = Exception  # type: ignore[misc, assignment]
    TableClient = None  # type: ignore[misc, assignment]
    UpdateMode = None  # type: ignore[misc, assignment]
    DefaultAzureCredential = None  # type: ignore[misc, assignment]

logger = logging.getLogger("satellite_classifier.cache")


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute a deterministic SHA-256 hex digest from raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


class BasePredictionCache(abc.ABC):
    """Abstract base class for duplicate-image prediction caching."""

    @abc.abstractmethod
    def get(self, image_hash: str, model_version: str) -> list[float] | None:
        """Retrieve cached probability vector, or None if not found."""
        raise NotImplementedError

    @abc.abstractmethod
    def set(
        self,
        image_hash: str,
        model_version: str,
        probabilities: list[float],
        predicted_class: str,
    ) -> bool:
        """Store probability vector in cache. Return True if successful."""
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all entries in the cache (primarily for tests/admin)."""
        raise NotImplementedError


class InMemoryPredictionCache(BasePredictionCache):
    """In-memory dictionary cache with thread locking, ideal for testing."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def get(self, image_hash: str, model_version: str) -> list[float] | None:
        key = (model_version, image_hash)
        with self._lock:
            probs = self._store.get(key)
            if probs is not None:
                return list(probs)
            return None

    def set(
        self,
        image_hash: str,
        model_version: str,
        probabilities: list[float],
        predicted_class: str,
    ) -> bool:
        key = (model_version, image_hash)
        with self._lock:
            self._store[key] = [float(p) for p in probabilities]
        return True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class SQLitePredictionCache(BasePredictionCache):
    """Local SQLite cache with WAL mode for single-instance or local dev."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            cache_dir = Path(__file__).resolve().parent.parent / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(cache_dir / "predictions.db")
        else:
            self.db_path = str(db_path)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_cache (
                        image_hash TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        probabilities TEXT NOT NULL,
                        predicted_class TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL,
                        PRIMARY KEY (image_hash, model_version)
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_cache_lookup
                    ON prediction_cache (image_hash, model_version);
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning(f"Failed to initialize SQLite prediction cache: {exc}")

    def get(self, image_hash: str, model_version: str) -> list[float] | None:
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT probabilities FROM prediction_cache
                WHERE image_hash = ? AND model_version = ?;
                """,
                (image_hash, model_version),
            )
            row = cursor.fetchone()
            if row:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    UPDATE prediction_cache
                    SET last_accessed_at = ?
                    WHERE image_hash = ? AND model_version = ?;
                    """,
                    (now, image_hash, model_version),
                )
                conn.commit()
                return [float(p) for p in json.loads(row[0])]
            return None
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.warning(f"SQLite cache read error for hash {image_hash[:8]}: {exc}")
            return None

    def set(
        self,
        image_hash: str,
        model_version: str,
        probabilities: list[float],
        predicted_class: str,
    ) -> bool:
        try:
            conn = self._get_connection()
            now = datetime.now(timezone.utc).isoformat()
            probs_json = json.dumps([float(p) for p in probabilities])
            conn.execute(
                """
                INSERT OR REPLACE INTO prediction_cache
                (image_hash, model_version, probabilities, predicted_class, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (image_hash, model_version, probs_json, predicted_class, now, now),
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.warning(f"SQLite cache write error for hash {image_hash[:8]}: {exc}")
            return False

    def clear(self) -> None:
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM prediction_cache;")
            conn.commit()
        except sqlite3.Error as exc:
            logger.warning(f"SQLite cache clear error: {exc}")


class AzureTablePredictionCache(BasePredictionCache):
    """
    Azure Table Storage cache shared across all Azure Container App replicas.

    Entities in the table use:
      PartitionKey = model_version
      RowKey = image_hash
    """

    def __init__(
        self,
        table_name: str = "predictioncache",
        storage_account: str | None = None,
        connection_string: str | None = None,
        client_id: str | None = None,
    ) -> None:
        self.table_name = table_name
        self.storage_account = storage_account or os.getenv("AZURE_STORAGE_ACCOUNT")
        self.connection_string = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID")
        self._table_client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if not AZURE_SDK_AVAILABLE:
            logger.warning(
                "azure-data-tables or azure-identity not installed; "
                "AzureTablePredictionCache disabled."
            )
            return

        try:
            if self.connection_string:
                self._table_client = TableClient.from_connection_string(
                    conn_str=self.connection_string,
                    table_name=self.table_name,
                )
            elif self.storage_account:
                endpoint = f"https://{self.storage_account}.table.core.windows.net"
                credential = (
                    DefaultAzureCredential(
                        managed_identity_client_id=self.client_id
                    )
                    if self.client_id
                    else DefaultAzureCredential()
                )
                self._table_client = TableClient(
                    endpoint=endpoint,
                    table_name=self.table_name,
                    credential=credential,
                )
            else:
                logger.warning(
                    "AzureTablePredictionCache: Neither AZURE_STORAGE_ACCOUNT nor "
                    "AZURE_STORAGE_CONNECTION_STRING is set."
                )
                return

            # Ensure table exists
            try:
                self._table_client.create_table()
                logger.info(
                    f"Azure Table '{self.table_name}' created in account '{self.storage_account}'."
                )
            except ResourceExistsError:
                pass
            except (HttpResponseError, OSError) as exc:
                logger.debug(f"Table existence check result: {exc}")

        except (HttpResponseError, OSError, ValueError) as exc:
            logger.warning(f"Failed to initialize Azure TableClient: {exc}")
            self._table_client = None

    def get(self, image_hash: str, model_version: str) -> list[float] | None:
        if self._table_client is None:
            return None

        try:
            entity = self._table_client.get_entity(
                partition_key=model_version,
                row_key=image_hash,
            )
            raw_probs = entity.get("probabilities")
            if raw_probs:
                return [float(p) for p in json.loads(raw_probs)]
            return None
        except ResourceNotFoundError:
            return None
        except (HttpResponseError, OSError, ValueError, KeyError) as exc:
            logger.warning(
                f"Azure Table cache get error for {image_hash[:8]}: {exc}"
            )
            return None

    def set(
        self,
        image_hash: str,
        model_version: str,
        probabilities: list[float],
        predicted_class: str,
    ) -> bool:
        if self._table_client is None:
            return False

        try:
            now = datetime.now(timezone.utc).isoformat()
            entity = {
                "PartitionKey": model_version,
                "RowKey": image_hash,
                "probabilities": json.dumps([float(p) for p in probabilities]),
                "predicted_class": predicted_class,
                "created_at": now,
                "last_accessed_at": now,
            }
            self._table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)
            return True
        except (HttpResponseError, OSError, ValueError) as exc:
            logger.warning(
                f"Azure Table cache set error for {image_hash[:8]}: {exc}"
            )
            return False

    def clear(self) -> None:
        # Avoid clearing shared production tables unintentionally
        logger.warning(
            "Clear operation invoked on AzureTablePredictionCache; ignored for safety."
        )


def get_prediction_cache() -> BasePredictionCache:
    """
    Factory creating the appropriate cache implementation.

    Selection logic:
    1. If PREDICTION_CACHE_BACKEND == 'memory': return InMemoryPredictionCache()
    2. If PREDICTION_CACHE_BACKEND == 'sqlite': return SQLitePredictionCache()
    3. If PREDICTION_CACHE_BACKEND == 'azure_table' or AZURE_STORAGE_ACCOUNT is set:
       return AzureTablePredictionCache()
    4. Default: SQLitePredictionCache() for offline/local development.
    """
    backend_env = os.getenv("PREDICTION_CACHE_BACKEND", "").lower().strip()

    if backend_env == "memory":
        logger.info("Prediction cache initialized: InMemoryPredictionCache")
        return InMemoryPredictionCache()

    if backend_env == "azure_table" or (
        not backend_env
        and (
            os.getenv("AZURE_STORAGE_ACCOUNT")
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
    ):
        table_name = os.getenv("AZURE_TABLE_NAME", "predictioncache")
        logger.info(
            f"Prediction cache initialized: AzureTablePredictionCache (table: {table_name})"
        )
        return AzureTablePredictionCache(table_name=table_name)

    logger.info("Prediction cache initialized: SQLitePredictionCache (local)")
    return SQLitePredictionCache()

