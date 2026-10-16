"""
Database module for storing and querying waste classification history.
Uses SQLite for robust, zero-configuration local persistence.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import config

DB_FILE = config.DB_PATH


def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the classification history database schema."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS waste_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_source TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                top2_class TEXT,
                top2_confidence REAL,
                top3_class TEXT,
                top3_confidence REAL,
                user_feedback TEXT DEFAULT 'Unverified',
                user_corrected_class TEXT,
                notes TEXT
            )
        """)
        conn.commit()


def add_scan(
    image_source: str,
    predicted_class: str,
    confidence: float,
    top2_class: str = None,
    top2_confidence: float = None,
    top3_class: str = None,
    top3_confidence: float = None,
    user_feedback: str = "Unverified",
    notes: str = ""
) -> int:
    """Inserts a new classification record and returns the scan ID."""
    init_db()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO waste_scans (
                timestamp, image_source, predicted_class, confidence,
                top2_class, top2_confidence, top3_class, top3_confidence,
                user_feedback, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_iso, image_source, predicted_class, float(confidence),
            top2_class, float(top2_confidence) if top2_confidence is not None else None,
            top3_class, float(top3_confidence) if top3_confidence is not None else None,
            user_feedback, notes
        ))
        conn.commit()
        return cursor.lastrowid


def update_feedback(scan_id: int, feedback: str, corrected_class: str = None):
    """Updates user verification or correction for a classification record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE waste_scans
            SET user_feedback = ?, user_corrected_class = ?
            WHERE id = ?
        """, (feedback, corrected_class, scan_id))
        conn.commit()


def get_all_scans() -> pd.DataFrame:
    """Retrieves all classification history records as a pandas DataFrame."""
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM waste_scans ORDER BY id DESC", conn
        )
    return df


def get_stats() -> dict:
    """Computes summary analytics for the dashboard metrics cards."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM waste_scans")
        total_scans = cursor.fetchone()[0]

        if total_scans == 0:
            return {
                "total_scans": 0,
                "avg_confidence": 0.0,
                "most_common_class": "N/A",
                "recyclable_count": 0,
                "recyclable_rate": 0.0
            }

        cursor.execute("SELECT AVG(confidence) FROM waste_scans")
        avg_confidence = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT predicted_class, COUNT(*) as cnt
            FROM waste_scans
            GROUP BY predicted_class
            ORDER BY cnt DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        most_common = row[0] if row else "N/A"

        # Count recyclable items (plastics, paper, cardboard, glass, metal) vs organic/trash
        cursor.execute("""
            SELECT COUNT(*) FROM waste_scans
            WHERE predicted_class IN ('plastic', 'paper', 'cardboard', 'glass', 'metal')
        """)
        recyclable_count = cursor.fetchone()[0]
        recyclable_rate = (recyclable_count / total_scans) * 100.0 if total_scans > 0 else 0.0

        return {
            "total_scans": total_scans,
            "avg_confidence": round(avg_confidence * 100, 1),
            "most_common_class": most_common,
            "recyclable_count": recyclable_count,
            "recyclable_rate": round(recyclable_rate, 1)
        }


def clear_history():
    """Deletes all scan history records."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM waste_scans")
        conn.commit()


def export_csv_data() -> str:
    """Exports history as a CSV string."""
    df = get_all_scans()
    return df.to_csv(index=False)
