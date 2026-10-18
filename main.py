
from pathlib import Path
import json
import sqlite3
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "edulens_final_dataset.txt"
DB_FILE = BASE_DIR / "edulens.db"

app = FastAPI(
    title="EduGuard AI",
    description="School intelligence, inspection prioritization and corrective-action tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_COLUMNS = [
    "school_id", "school_name", "district", "management_type",
    "total_teachers", "total_enrollment", "inspection_priority_score",
    "inspection_priority_category", "recommended_follow_up",
]

def load_dataset():
    if not DATA_FILE.exists():
        raise RuntimeError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")

    df["school_id"] = df["school_id"].astype(str).str.strip()
    return df

schools = load_dataset()

def clean(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)

def record(row):
    return {str(k): clean(v) for k, v in row.items()}

def connect_db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def log_event(con, case_id, event, details):
    con.execute(
        """INSERT INTO audit_log
           (case_id, event, details, created_at)
           VALUES (?, ?, ?, ?)""",
        (case_id, event, json.dumps(details), now()),
    )

def init_db():
    with connect_db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            school_id TEXT,
            school_name TEXT,
            category TEXT NOT NULL,
            issue TEXT NOT NULL,
            action_required TEXT DEFAULT '',
            assigned_to TEXT DEFAULT '',
            due_date TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'Open',
            evidence_notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            event TEXT NOT NULL,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

init_db()

class CaseCreate(BaseModel):
    school_id: Optional[str] = None
    category: str = Field(min_length=1, max_length=100)
    issue: str = Field(min_length=3, max_length=2000)
    action_required: str = Field(default="", max_length=2000)
    assigned_to: str = Field(default="", max_length=200)
    due_date: str = Field(default="", max_length=30)

class CaseUpdate(BaseModel):
    category: Optional[str] = Field(default=None, max_length=100)
    issue: Optional[str] = Field(default=None, max_length=2000)
    action_required: Optional[str] = Field(default=None, max_length=2000)
    assigned_to: Optional[str] = Field(default=None, max_length=200)
    due_date: Optional[str] = Field(default=None, max_length=30)
    status: Optional[str] = None
    evidence_notes: Optional[str] = Field(default=None, max_length=4000)

ALLOWED_STATUSES = {
    "Open", "Under Review", "Action Required",
    "Pending Verification", "Closed"
}

@app.get("/")
def home():
    return {
        "app": "EduGuard AI",
        "status": "running",
        "message": "School intelligence API is ready",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "dataset_rows": len(schools),
        "database": str(DB_FILE.name),
    }

@app.get("/api/summary")
def summary():
    df = schools
    districts = int(df["district"].nunique())
    result = {
        "total_schools": int(len(df)),
        "districts": districts,
        "data_updated_field": "last_data_updated",
        "priority_categories": {},
        "management_types": {},
        "data_quality_counts": {},
    }

    for column, key in [
        ("inspection_priority_category", "priority_categories"),
        ("management_type", "management_types"),
        ("data_quality_status", "data_quality_counts"),
    ]:
        if column in df.columns:
            result[key] = {
                str(k): int(v)
                for k, v in df[column].fillna("Unknown").value_counts().items()
            }

    if "missing_critical_data_count" in df.columns:
        missing = pd.to_numeric(
            df["missing_critical_data_count"], errors="coerce"
        )
        result["schools_with_missing_critical_data"] = int(
            (missing.fillna(0) > 0).sum()
        )

    with connect_db() as con:
        result["open_cases"] = con.execute(
            "SELECT COUNT(*) FROM cases WHERE status != 'Closed'"
        ).fetchone()[0]

    return result

@app.get("/api/schools")
def list_schools(
    search: str = Query(default="", max_length=200),
    district: str = Query(default="", max_length=200),
    category: str = Query(default="", max_length=100),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    df = schools

    if search:
        mask = (
            df["school_name"].astype(str).str.contains(search, case=False, na=False, regex=False)
            | df["school_id"].astype(str).str.contains(search, case=False, na=False, regex=False)
        )
        df = df[mask]

    if district:
        df = df[df["district"].astype(str).str.casefold() == district.casefold()]

    if category and "inspection_priority_category" in df.columns:
        df = df[
            df["inspection_priority_category"].astype(str).str.casefold()
            == category.casefold()
        ]

    total = int(len(df))

    if "inspection_priority_score" in df.columns:
        df = df.assign(
            _sort=pd.to_numeric(df["inspection_priority_score"], errors="coerce")
        ).sort_values("_sort", ascending=False, na_position="last")

    page = df.iloc[offset:offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [record(row) for _, row in page.iterrows()],
    }

@app.get("/api/schools/{school_id}")
def school_detail(school_id: str):
    matches = schools[schools["school_id"] == school_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="School not found")

    school = record(matches.iloc[0].to_dict())
    with connect_db() as con:
        cases = con.execute(
            "SELECT * FROM cases WHERE school_id = ? ORDER BY id DESC",
            (school_id,),
        ).fetchall()

    school["cases"] = [dict(c) for c in cases]
    return school

@app.get("/api/districts")
def districts():
    return sorted(
        str(x) for x in schools["district"].dropna().unique()
    )

@app.get("/api/cases")
def list_cases(status: str = "", school_id: str = ""):
    sql = "SELECT * FROM cases WHERE 1=1"
    params = []

    if status:
        sql += " AND status = ?"
        params.append(status)
    if school_id:
        sql += " AND school_id = ?"
        params.append(school_id)

    sql += " ORDER BY id DESC"

    with connect_db() as con:
        rows = con.execute(sql, params).fetchall()
    return {"items": [dict(row) for row in rows]}

@app.post("/api/cases", status_code=201)
def create_case(data: CaseCreate):
    school_name = None

    if data.school_id:
        match = schools[schools["school_id"] == data.school_id]
        if match.empty:
            raise HTTPException(status_code=404, detail="School ID not found")
        school_name = str(match.iloc[0]["school_name"])

    timestamp = now()
    with connect_db() as con:
        cursor = con.execute(
            """INSERT INTO cases
            (school_id, school_name, category, issue, action_required,
             assigned_to, due_date, status, evidence_notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Open', '', ?, ?)""",
            (
                data.school_id, school_name, data.category, data.issue,
                data.action_required, data.assigned_to, data.due_date,
                timestamp, timestamp,
            ),
        )
        case_id = cursor.lastrowid
        log_event(con, case_id, "Case created", data.model_dump())
        row = con.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()

    return dict(row)

@app.patch("/api/cases/{case_id}")
def update_case(case_id: int, data: CaseUpdate):
    changes = data.model_dump(exclude_unset=True, exclude_none=True)

    if not changes:
        raise HTTPException(status_code=400, detail="No updates supplied")

    if "status" in changes and changes["status"] not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {sorted(ALLOWED_STATUSES)}",
        )

    with connect_db() as con:
        existing = con.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()

        if existing is None:
            raise HTTPException(status_code=404, detail="Case not found")

        changes["updated_at"] = now()
        assignments = ", ".join(f"{key} = ?" for key in changes)
        values = list(changes.values()) + [case_id]

        con.execute(
            f"UPDATE cases SET {assignments} WHERE id = ?", values
        )
        log_event(con, case_id, "Case updated", changes)

        updated = con.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()

    return dict(updated)

@app.get("/api/audit")
def audit_log(case_id: Optional[int] = None):
    with connect_db() as con:
        if case_id is None:
            rows = con.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM audit_log WHERE case_id = ? ORDER BY id DESC",
                (case_id,),
            ).fetchall()

    return {"items": [dict(row) for row in rows]}

app.mount("/app", StaticFiles(directory=BASE_DIR / "static", html=True), name="frontend")

