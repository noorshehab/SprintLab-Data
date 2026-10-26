#this script creates the indexes needed for question lookup
import argparse

try:
    from .db_config import DBConfig, add_arguments, config_from_namespace
except ImportError:  # direct script execution
    from db_config import DBConfig, add_arguments, config_from_namespace


def create_indexes(host: str | None = None, user: str | None = None,
                   password: str | None = None, port: int | str | None = None,
                   db_name: str | None = None):
    cfg = DBConfig.from_values(
        host=host, user=user, password=password, port=port, db_name=db_name)
    db_name = cfg.db_name

    print(f"Connecting to MySQL at {cfg.host}:{cfg.port}/{db_name}...")
    conn = cfg.connect()

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

def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(description="Create SprintLab MySQL indexes"))
    args = parser.parse_args(argv)
    create_indexes(**vars(config_from_namespace(args)))


if __name__ == "__main__":
    main()