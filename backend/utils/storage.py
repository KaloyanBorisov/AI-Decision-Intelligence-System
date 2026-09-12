"""
Persistent SQL storage for users and API keys.
Supports PostgreSQL (production on Render) and SQLite (local development).
Note: Render Free PostgreSQL instances expire 90 days after creation
(~Dec 4, 2026 if created Sep 5, 2026).
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class UserModel(Base):
    """SQLAlchemy model for persistent user accounts."""

    __tablename__ = "users"

    id = Column(String(128), primary_key=True)
    username = Column(String(128), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(64), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    last_login = Column(DateTime, nullable=True)
    extra_data = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert database record to dictionary matching auth and schema expectations."""
        d = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "hashed_password": self.hashed_password,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "failed_login_attempts": self.failed_login_attempts,
            "last_login": self.last_login,
        }
        if self.extra_data:
            try:
                extra = json.loads(self.extra_data)
                if isinstance(extra, dict):
                    # Don't overwrite primary fields with extra_data
                    for k, v in extra.items():
                        if k not in d:
                            d[k] = v
            except Exception:
                pass
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
        """Build UserModel from dictionary."""
        known_keys = {
            "id",
            "username",
            "email",
            "hashed_password",
            "role",
            "is_active",
            "is_verified",
            "created_at",
            "updated_at",
            "failed_login_attempts",
            "last_login",
        }

        def parse_dt(val):
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    return datetime.utcnow()
            elif isinstance(val, datetime):
                return val
            return None

        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            id=str(data["id"]),
            username=str(data["username"]),
            email=str(data["email"]),
            hashed_password=str(data["hashed_password"]),
            role=str(data.get("role", "user")),
            is_active=bool(data.get("is_active", True)),
            is_verified=bool(data.get("is_verified", False)),
            created_at=parse_dt(data.get("created_at")) or datetime.utcnow(),
            updated_at=parse_dt(data.get("updated_at")) or datetime.utcnow(),
            failed_login_attempts=int(data.get("failed_login_attempts", 0) or 0),
            last_login=parse_dt(data.get("last_login")),
            extra_data=json.dumps(extra) if extra else None,
        )


class APIKeyModel(Base):
    """SQLAlchemy model for persistent API keys."""

    __tablename__ = "api_keys"

    key_hash = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, key_hash: str, data: Dict[str, Any]) -> "APIKeyModel":
        def parse_dt(val):
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    return None
            elif isinstance(val, datetime):
                return val
            return None

        return cls(
            key_hash=str(key_hash),
            user_id=str(data["user_id"]),
            name=str(data.get("name", "Default Key")),
            created_at=parse_dt(data.get("created_at")) or datetime.utcnow(),
            expires_at=parse_dt(data.get("expires_at")),
            is_active=bool(data.get("is_active", True)),
        )


class DatasetModel(Base):
    """SQLAlchemy model for persistent uploaded-dataset metadata."""

    __tablename__ = "datasets"

    id = Column(String(128), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    size = Column(Integer, default=0, nullable=False)
    rows = Column(Integer, default=0, nullable=False)
    columns = Column(Integer, default=0, nullable=False)
    column_names = Column(Text, nullable=True)  # JSON-encoded list[str]

    def to_dict(self) -> Dict[str, Any]:
        try:
            column_names = json.loads(self.column_names) if self.column_names else []
        except Exception:
            column_names = []
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "size": self.size,
            "rows": self.rows,
            "columns": self.columns,
            "column_names": column_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetModel":
        def parse_dt(val):
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    return datetime.utcnow()
            elif isinstance(val, datetime):
                return val
            return datetime.utcnow()

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=data.get("description"),
            file_path=str(data["file_path"]),
            created_at=parse_dt(data.get("created_at")),
            size=int(data.get("size", 0) or 0),
            rows=int(data.get("rows", 0) or 0),
            columns=int(data.get("columns", 0) or 0),
            column_names=json.dumps(data.get("column_names") or []),
        )


def _get_engine(database_url: Optional[str] = None):
    """Configure SQLAlchemy engine supporting PostgreSQL and SQLite."""
    url = database_url or settings.database_url

    if url:
        # Render and Heroku use 'postgres://' which SQLAlchemy 1.4+ rejects;
        # normalize to 'postgresql://'
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)

        logger.info(f"Connecting to configured database: {url.split('@')[-1]}")
        if url.startswith("sqlite"):
            return create_engine(url, connect_args={"check_same_thread": False})
        else:
            return create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=300,
            )
    else:
        # Fallback to local SQLite database in storage/ directory
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)
        db_path = storage_dir / "decisera.db"
        sqlite_url = f"sqlite:///{db_path.resolve()}"
        logger.info(
            f"No DATABASE_URL configured. Using local SQLite database: {db_path}"
        )
        return create_engine(sqlite_url, connect_args={"check_same_thread": False})


class SQLUserStorage:
    """Dict-compatible wrapper around SQLAlchemy for user persistence."""

    def __init__(self, engine=None):
        self.engine = engine or _get_engine()
        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))

    def _get_session(self):
        return self.SessionFactory()

    def get(self, user_id: str, default=None) -> Optional[Dict[str, Any]]:
        session = self._get_session()
        try:
            user = session.get(UserModel, str(user_id))
            if user:
                return user.to_dict()
            return default
        finally:
            session.close()

    def __getitem__(self, user_id: str) -> Dict[str, Any]:
        val = self.get(user_id)
        if val is None:
            raise KeyError(user_id)
        return val

    def __setitem__(self, user_id: str, data: Dict[str, Any]):
        session = self._get_session()
        try:
            data = dict(data)
            data["id"] = user_id
            existing = session.get(UserModel, str(user_id))
            if existing:
                # Update fields
                user_obj = UserModel.from_dict(data)
                existing.username = user_obj.username
                existing.email = user_obj.email
                existing.hashed_password = user_obj.hashed_password
                existing.role = user_obj.role
                existing.is_active = user_obj.is_active
                existing.is_verified = user_obj.is_verified
                existing.failed_login_attempts = user_obj.failed_login_attempts
                existing.last_login = user_obj.last_login
                existing.updated_at = datetime.utcnow()
                existing.extra_data = user_obj.extra_data
            else:
                user_obj = UserModel.from_dict(data)
                session.add(user_obj)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __delitem__(self, user_id: str):
        session = self._get_session()
        try:
            existing = session.get(UserModel, str(user_id))
            if not existing:
                raise KeyError(user_id)
            session.delete(existing)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __contains__(self, user_id: str) -> bool:
        session = self._get_session()
        try:
            return session.get(UserModel, str(user_id)) is not None
        finally:
            session.close()

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        session = self._get_session()
        try:
            users = session.query(UserModel).all()
            return [(u.id, u.to_dict()) for u in users]
        finally:
            session.close()

    def values(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            users = session.query(UserModel).all()
            return [u.to_dict() for u in users]
        finally:
            session.close()

    def keys(self) -> List[str]:
        session = self._get_session()
        try:
            ids = session.query(UserModel.id).all()
            return [row[0] for row in ids]
        finally:
            session.close()

    def clear(self):
        session = self._get_session()
        try:
            session.query(UserModel).delete()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __len__(self) -> int:
        session = self._get_session()
        try:
            return session.query(UserModel.id).count()
        finally:
            session.close()


class SQLAPIKeyStorage:
    """Dict-compatible wrapper around SQLAlchemy for API key persistence."""

    def __init__(self, engine=None):
        self.engine = engine or _get_engine()
        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))

    def _get_session(self):
        return self.SessionFactory()

    def get(self, key_hash: str, default=None) -> Optional[Dict[str, Any]]:
        session = self._get_session()
        try:
            key_obj = session.get(APIKeyModel, str(key_hash))
            if key_obj:
                return key_obj.to_dict()
            return default
        finally:
            session.close()

    def __getitem__(self, key_hash: str) -> Dict[str, Any]:
        val = self.get(key_hash)
        if val is None:
            raise KeyError(key_hash)
        return val

    def __setitem__(self, key_hash: str, data: Dict[str, Any]):
        session = self._get_session()
        try:
            existing = session.get(APIKeyModel, str(key_hash))
            if existing:
                key_obj = APIKeyModel.from_dict(key_hash, data)
                existing.name = key_obj.name
                existing.expires_at = key_obj.expires_at
                existing.is_active = key_obj.is_active
            else:
                key_obj = APIKeyModel.from_dict(key_hash, data)
                session.add(key_obj)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __delitem__(self, key_hash: str):
        session = self._get_session()
        try:
            existing = session.get(APIKeyModel, str(key_hash))
            if not existing:
                raise KeyError(key_hash)
            session.delete(existing)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __contains__(self, key_hash: str) -> bool:
        session = self._get_session()
        try:
            return session.get(APIKeyModel, str(key_hash)) is not None
        finally:
            session.close()

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        session = self._get_session()
        try:
            keys = session.query(APIKeyModel).all()
            return [(k.key_hash, k.to_dict()) for k in keys]
        finally:
            session.close()

    def values(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            keys = session.query(APIKeyModel).all()
            return [k.to_dict() for k in keys]
        finally:
            session.close()

    def keys(self) -> List[str]:
        session = self._get_session()
        try:
            keys = session.query(APIKeyModel.key_hash).all()
            return [row[0] for row in keys]
        finally:
            session.close()

    def clear(self):
        session = self._get_session()
        try:
            session.query(APIKeyModel).delete()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __len__(self) -> int:
        session = self._get_session()
        try:
            return session.query(APIKeyModel.key_hash).count()
        finally:
            session.close()


class SQLDatasetStorage:
    """SQLAlchemy-backed persistence for uploaded-dataset metadata."""

    def __init__(self, engine=None):
        self.engine = engine or _get_engine()
        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))

    def _get_session(self):
        return self.SessionFactory()

    def add(self, data: Dict[str, Any]) -> None:
        session = self._get_session()
        try:
            session.add(DatasetModel.from_dict(data))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        session = self._get_session()
        try:
            obj = session.get(DatasetModel, str(dataset_id))
            return obj.to_dict() if obj else None
        finally:
            session.close()

    def update(self, dataset_id: str, data: Dict[str, Any]) -> None:
        session = self._get_session()
        try:
            existing = session.get(DatasetModel, str(dataset_id))
            if not existing:
                raise KeyError(dataset_id)
            merged = existing.to_dict()
            merged.update(data)
            updated = DatasetModel.from_dict(merged)
            existing.name = updated.name
            existing.description = updated.description
            existing.file_path = updated.file_path
            existing.size = updated.size
            existing.rows = updated.rows
            existing.columns = updated.columns
            existing.column_names = updated.column_names
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete(self, dataset_id: str) -> bool:
        session = self._get_session()
        try:
            existing = session.get(DatasetModel, str(dataset_id))
            if not existing:
                return False
            session.delete(existing)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def all(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            objs = session.query(DatasetModel).order_by(DatasetModel.created_at).all()
            return [o.to_dict() for o in objs]
        finally:
            session.close()

    def __len__(self) -> int:
        session = self._get_session()
        try:
            return session.query(DatasetModel.id).count()
        finally:
            session.close()


class ModelMetadataModel(Base):
    """SQLAlchemy model for persistent trained-model metadata.

    Only JSON-serializable metadata lives here. The actual fitted model
    (and its AutoML wrapper / SHAP sample) are dumped to disk via joblib;
    this row just records where to find them so they can be lazily
    reloaded after a restart.
    """

    __tablename__ = "models"

    model_id = Column(String(128), primary_key=True)
    dataset_id = Column(String(128), nullable=True)
    target_column = Column(String(255), nullable=True)
    task_type = Column(String(64), nullable=True)
    best_model_name = Column(String(255), nullable=True)
    best_score = Column(Float, default=0.0, nullable=False)
    feature_names = Column(Text, nullable=True)  # JSON-encoded list[str]
    all_results = Column(Text, nullable=True)  # JSON-encoded dict
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    automl_path = Column(String(1024), nullable=True)
    model_path = Column(String(1024), nullable=True)
    xsample_path = Column(String(1024), nullable=True)
    # Which MLflow experiment this model was trained under, so inference
    # traces can be routed back to it (see ModelService._use_experiment_for_tracing).
    mlflow_experiment_name = Column(String(255), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        def load_json(val, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except Exception:
                return default

        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id or "",
            "target_column": self.target_column or "",
            "task_type": self.task_type or "classification",
            "best_model_name": self.best_model_name or "AutoML",
            "best_score": self.best_score or 0.0,
            "feature_names": load_json(self.feature_names, []),
            "all_results": load_json(self.all_results, {}),
            "created_at": self.created_at,
            "automl_path": self.automl_path or "",
            "model_path": self.model_path or "",
            "xsample_path": self.xsample_path or "",
            "mlflow_experiment_name": self.mlflow_experiment_name or "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadataModel":
        def parse_dt(val):
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    return datetime.utcnow()
            elif isinstance(val, datetime):
                return val
            return datetime.utcnow()

        return cls(
            model_id=str(data["model_id"]),
            dataset_id=str(data.get("dataset_id") or ""),
            target_column=str(data.get("target_column") or ""),
            task_type=str(data.get("task_type") or "classification"),
            best_model_name=str(data.get("best_model_name") or "AutoML"),
            best_score=float(data.get("best_score", 0.0) or 0.0),
            feature_names=json.dumps(data.get("feature_names") or []),
            all_results=json.dumps(data.get("all_results") or {}, default=str),
            created_at=parse_dt(data.get("created_at")),
            automl_path=str(data.get("automl_path") or ""),
            model_path=str(data.get("model_path") or ""),
            xsample_path=str(data.get("xsample_path") or ""),
            mlflow_experiment_name=str(data.get("mlflow_experiment_name") or ""),
        )


class SQLModelStorage:
    """SQLAlchemy-backed persistence for trained-model metadata."""

    def __init__(self, engine=None):
        self.engine = engine or _get_engine()
        Base.metadata.create_all(bind=self.engine)
        self._migrate_columns()
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))

    def _migrate_columns(self) -> None:
        """Lightweight migration: create_all only creates missing tables, not
        columns added to a model after the table already existed on disk.
        Add those columns in place (SQLite and Postgres both support a bare
        ALTER TABLE ... ADD COLUMN)."""
        try:
            inspector = inspect(self.engine)
            if "models" not in inspector.get_table_names():
                return
            existing_cols = {c["name"] for c in inspector.get_columns("models")}
            if "mlflow_experiment_name" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(
                        text("ALTER TABLE models ADD COLUMN mlflow_experiment_name VARCHAR(255)")
                    )
                logger.info("Migrated models table: added mlflow_experiment_name column")
        except Exception as exc:
            logger.warning(f"Could not migrate models table: {exc}")

    def _get_session(self):
        return self.SessionFactory()

    def upsert(self, model_id: str, data: Dict[str, Any]) -> None:
        session = self._get_session()
        try:
            data = dict(data)
            data["model_id"] = model_id
            existing = session.get(ModelMetadataModel, str(model_id))
            new_obj = ModelMetadataModel.from_dict(data)
            if existing:
                existing.dataset_id = new_obj.dataset_id
                existing.target_column = new_obj.target_column
                existing.task_type = new_obj.task_type
                existing.best_model_name = new_obj.best_model_name
                existing.best_score = new_obj.best_score
                existing.feature_names = new_obj.feature_names
                existing.all_results = new_obj.all_results
                existing.automl_path = new_obj.automl_path
                existing.model_path = new_obj.model_path
                existing.xsample_path = new_obj.xsample_path
                existing.mlflow_experiment_name = new_obj.mlflow_experiment_name
            else:
                session.add(new_obj)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        session = self._get_session()
        try:
            obj = session.get(ModelMetadataModel, str(model_id))
            return obj.to_dict() if obj else None
        finally:
            session.close()

    def delete(self, model_id: str) -> bool:
        session = self._get_session()
        try:
            existing = session.get(ModelMetadataModel, str(model_id))
            if not existing:
                return False
            session.delete(existing)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def all(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            objs = (
                session.query(ModelMetadataModel)
                .order_by(ModelMetadataModel.created_at)
                .all()
            )
            return [o.to_dict() for o in objs]
        finally:
            session.close()

    def __len__(self) -> int:
        session = self._get_session()
        try:
            return session.query(ModelMetadataModel.model_id).count()
        finally:
            session.close()


# Legacy JSONStorage preserved for backwards compatibility
class JSONStorage:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lock = threading.Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                try:
                    self._data = json.load(f)
                except json.JSONDecodeError:
                    self._data = {}
        else:
            self._data = {}
            self._save()

    def _save(self):
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.file_path, "w") as f:
            json.dump(self._data, f, indent=2, default=json_serial)

    def get(self, key: str, default=None):
        with self.lock:
            return self._data.get(key, default)

    def __setitem__(self, key: str, value: Any):
        with self.lock:
            self._data[key] = value
            self._save()

    def __getitem__(self, key: str):
        with self.lock:
            return self._data[key]

    def __delitem__(self, key: str):
        with self.lock:
            del self._data[key]
            self._save()

    def __contains__(self, key: str):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def clear(self):
        with self.lock:
            self._data = {}
            self._save()

    def __len__(self):
        return len(self._data)


# Shared engine for user and API key storage
_default_engine = _get_engine()

# Initialize primary persistent storage
users_storage = SQLUserStorage(_default_engine)
api_keys_storage = SQLAPIKeyStorage(_default_engine)
datasets_storage = SQLDatasetStorage(_default_engine)
models_storage = SQLModelStorage(_default_engine)

# One-time automatic migration: if legacy users.json exists and SQL is empty
try:
    legacy_users_json = Path("storage/users.json")
    if legacy_users_json.exists() and len(users_storage) == 0:
        with open(legacy_users_json, "r") as f:
            legacy_data = json.load(f)
            if isinstance(legacy_data, dict):
                logger.info(
                    f"Migrating {len(legacy_data)} users from storage/users.json "
                    "to SQL database..."
                )
                for uid, udata in legacy_data.items():
                    if (
                        isinstance(udata, dict)
                        and "username" in udata
                        and "hashed_password" in udata
                    ):
                        users_storage[uid] = udata
                logger.info("✓ Legacy users migration completed.")
except Exception as migration_err:
    logger.warning(f"Could not check or migrate legacy users.json: {migration_err}")
