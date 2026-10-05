from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./prediction_logs.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    event = Column(String(50))
    request_id = Column(String(100))
    mode = Column(String(50))
    filename = Column(String(255))

    # Exact uploaded-file identifier.
    # The image itself is NOT stored.
    image_hash = Column(String(64), index=True)

    predicted_class = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_percent = Column(Float, nullable=True)
    p_unknown = Column(Float, nullable=True)

    status = Column(String(50), nullable=True)
    final_result = Column(String(100), nullable=True)
    threshold = Column(Float, nullable=True)

    processing_time_ms = Column(Float, nullable=True)

    # True when the result came from the SQL cache
    # instead of running the model.
    cache_hit = Column(Boolean, default=False)

    error = Column(String(1000), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)