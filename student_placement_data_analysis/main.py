import os
from pathlib import Path

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

COHORT_YEAR = int(os.getenv("COHORT_YEAR", "2027"))


def connect_db():
    """Connect to the SQLite database."""
    db_path = Path(__file__).resolve().parent.parent / "placement.db"
    return sqlite3.connect(db_path)


def load_data(connection):
    """Load only the columns needed for analysis."""
    students = pd.read_sql(
        "SELECT student_id, department, graduation_year, eligible_for_placement "
        "FROM students",
        connection,
    )
    placements = pd.read_sql(
        "SELECT placement_id, student_id, company, role, package_lpa, "
        "placement_date, status FROM placements",
        connection,
    )
    student_skills = pd.read_sql(
        "SELECT ss.student_id, sk.skill_name "
        "FROM student_skills ss JOIN skills sk ON ss.skill_id = sk.skill_id",
        connection,
    )
    return students, placements, student_skills

def remove_duplicate_offers(placements):
    """
    Guard against duplicate rows in imported data.
    The database also enforces uniqueness on student + company + role + date.
    """
    key = ["student_id", "company", "role", "placement_date"]
    return placements.drop_duplicates(subset=key, keep="first").copy()

def department_placement_report(students, placements):
    cohort = students[
        (students["graduation_year"] == COHORT_YEAR)
        & (students["eligible_for_placement"] == 1)
    ].copy()

    accepted_ids = set(
        placements.loc[placements["status"] == "Accepted", "student_id"].unique()
    )
    cohort["is_placed"] = cohort["student_id"].isin(accepted_ids)

    report = (
        cohort.groupby("department", as_index=False)
        .agg(
            eligible_students=("student_id", "nunique"),
            placed_students=("is_placed", "sum"),
        )
    )
    report["placement_rate_percent"] = np.where(
        report["eligible_students"] > 0,
        report["placed_students"] / report["eligible_students"] * 100,
        0,
    ).round(2)
    return report.sort_values("department")

def package_summary(placements):
    accepted = placements[placements["status"] == "Accepted"].copy()
    accepted["package_lpa"] = pd.to_numeric(accepted["package_lpa"], errors="coerce")
    packages = accepted["package_lpa"].dropna().to_numpy(dtype=float)

    if packages.size == 0:
        return pd.DataFrame([{
            "accepted_offers": 0, "mean_lpa": np.nan, "median_lpa": np.nan,
            "minimum_lpa": np.nan, "maximum_lpa": np.nan
        }])

    return pd.DataFrame([{
        "accepted_offers": int(packages.size),
        "mean_lpa": round(float(np.mean(packages)), 2),
        "median_lpa": round(float(np.median(packages)), 2),
        "minimum_lpa": round(float(np.min(packages)), 2),
        "maximum_lpa": round(float(np.max(packages)), 2),
    }])

def skill_demand_report(student_skills):
    # Count students with each skill, not repeated occurrences of the same skill.
    unique_pairs = student_skills.drop_duplicates(["student_id", "skill_name"])
    report = (
        unique_pairs.groupby("skill_name", as_index=False)
        .agg(students_with_skill=("student_id", "nunique"))
        .sort_values(["students_with_skill", "skill_name"], ascending=[False, True])
    )
    return report

def save_charts(department_report, placements, skills_report):
    if not department_report.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(department_report["department"], department_report["placement_rate_percent"])
        plt.title(f"Placement Rate by Department ({COHORT_YEAR})")
        plt.xlabel("Department")
        plt.ylabel("Placement rate (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "placement_rate_by_department.png", dpi=150)
        plt.close()

    accepted = placements.loc[placements["status"] == "Accepted", "package_lpa"].dropna()
    if not accepted.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(accepted, bins=min(8, max(3, len(accepted))), edgecolor="black")
        plt.title("Accepted Offer Package Distribution")
        plt.xlabel("Annual package (LPA)")
        plt.ylabel("Number of offers")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "accepted_package_distribution.png", dpi=150)
        plt.close()

    top_skills = skills_report.head(10).sort_values("students_with_skill")
    if not top_skills.empty:
        plt.figure(figsize=(8, 5))
        plt.barh(top_skills["skill_name"], top_skills["students_with_skill"])
        plt.title("Most Common Student Skills")
        plt.xlabel("Number of students")
        plt.ylabel("Skill")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "top_skills.png", dpi=150)
        plt.close()


def main():
    connection = None

    try:
        connection = connect_db()
        students, placements, student_skills = load_data(connection)

    except sqlite3.Error as error:
        print("Could not connect to SQLite database or load tables.")
        print(f"SQLite error: {error}")
        return

    finally:
        if connection is not None:
            connection.close()



    placements = remove_duplicate_offers(placements)

    department_report = department_placement_report(students, placements)
    packages_report = package_summary(placements)
    skills_report = skill_demand_report(student_skills)

    department_report.to_csv(REPORT_DIR / "department_placement_report.csv", index=False)
    packages_report.to_csv(REPORT_DIR / "package_summary.csv", index=False)
    skills_report.to_csv(REPORT_DIR / "skill_demand_report.csv", index=False)
    save_charts(department_report, placements, skills_report)

    # Aggregate-only output: no names, roll numbers, or student-level rows.
    print(f"\nSTUDENT PLACEMENT SUMMARY — COHORT {COHORT_YEAR}")
    print("\nDepartment-wise placement report:")
    print(department_report.to_string(index=False))
    print("\nAccepted offer package summary (annual CTC in LPA):")
    print(packages_report.to_string(index=False))
    print("\nSkill demand report:")
    print(skills_report.to_string(index=False))
    print(f"\nReports and charts saved in: {REPORT_DIR.resolve()}")

if __name__ == "__main__":
    main()
