#this script creates the indexes needed for question lookup
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def create_indexes():
    host = os.getenv("DB_HOST", "localhost")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "password")
    port = int(os.getenv("DB_PORT", "3306"))
    db_name = os.getenv("DB_NAME", "sprintlab_db")

    print(f"Connecting to MySQL at {host}:{port}/{db_name}...")
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port,
        database=db_name,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

    # Tuple format: (table_name, index_name, columns_sql)
    indexes = [
        # Curriculum lookups (from yellow sticky note + SCD2 relationships)
        ("Unit", "idx_unit_unitid", "`UnitID`"),
        ("Lesson", "idx_lesson_lessonid", "`LessonID`"),
        ("Lesson", "idx_lesson_unitid", "`UnitID`"),
        ("LearningObjective", "idx_learningobjective_id", "`LearningObjectiveID`"),
        ("LearningObjective", "idx_learningobjective_lessonid", "`LessonID`"),
        ("SkillCluster", "idx_skillcluster_lessonid", "`LessonID`"),
        ("SkillCluster", "idx_skillcluster_unitid", "`UnitID`"),
        ("SkillCluster", "idx_skillcluster_objectiveid", "`LearningObjectiveID`"),

        # Question, Options & Diagnostics indexes
        ("QuestionContent", "idx_questioncontent_skillcluster", "`SkillClusterID`"),
        ("Options", "idx_options_questionid", "`QuestionID`"),
        ("QuestionDiagnostics", "idx_qdiag_questionid", "`QuestionID`"),
        (
            "QuestionDiagnostics",
            "idx_qdiag_treatment_search",
            "`Current`, `LanguageLevel`, `CognitiveLoadIndex`, `BloomTax`, `VisualDependancy`, `LogicalSteps`, `MultiConceptFlag`, `TheoreticalTime`"
        ),

        # Gameplay & Student indexes
        ("Response", "idx_response_match_student", "`MatchID`, `StudentID`"),
        ("Response", "idx_response_question", "`QuestionID`"),
        ("StudentClass", "idx_studentclass_classid", "`ClassID`"),
        ("StudentMasteryLevels", "idx_masterylevels_skillcluster", "`SkillClusterID`"),
        ("StudentMasteryLevels", "idx_masterylevels_student_current", "`StudentID`, `Current`"),

        # Diagnostics & Treatments
        ("DiagnosticThresholds", "idx_diagthresh_diagnosisid", "`DiagnosisID`"),
        ("StudentDiagnostics", "idx_studentdiag_student_current", "`StudentID`, `Current`"),
        ("Treatment", "idx_treatment_student_diag", "`StudentID`, `DiagnosisID`"),
    ]

    try:
        with conn.cursor() as cursor:
            for table_name, index_name, columns in indexes:
                cursor.execute(
                    """
                    SELECT COUNT(1) AS cnt
                    FROM information_schema.statistics
                    WHERE table_schema = %s
                      AND table_name = %s
                      AND index_name = %s;
                    """,
                    (db_name, table_name, index_name),
                )
                exists = cursor.fetchone()["cnt"] > 0

                if not exists:
                    print(f"Creating index `{index_name}` on `{table_name}` ({columns})...")
                    cursor.execute(f"CREATE INDEX `{index_name}` ON `{table_name}` ({columns});")
                else:
                    print(f"Index `{index_name}` on `{table_name}` already exists. Skipping.")

        print("\nAll indexes verified and created successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    create_indexes()