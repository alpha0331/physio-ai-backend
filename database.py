from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./physio_ai.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FlaggedRep(Base):
    __tablename__ = "flagged_reps"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    exercise = Column(String)
    rep_number = Column(Integer)
    issue = Column(String)
    image_path = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Creates the table(s) if they don't exist yet
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
