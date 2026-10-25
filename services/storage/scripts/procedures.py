import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def create_scd2_stored_procedures():
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = int(os.getenv("DB_PORT"))
    db_name = os.getenv("DB_NAME")

    print(f"Connecting to MySQL server at {host}:{port}/{db_name}...")
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port,
        database=db_name,
        autocommit=True,
    )

    procedures = {
        # 1. Update Unit Version (SCD Type 2)
        "sp_update_unit_scd2": """
        CREATE PROCEDURE sp_update_unit_scd2(
            IN p_unit_id VARCHAR(64),
            IN p_language VARCHAR(16),
            IN p_curriculum VARCHAR(64),
            IN p_grade VARCHAR(32),
            IN p_unit_name VARCHAR(255),
            IN p_unit_content TEXT
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Expire the currently active version
            UPDATE Unit
            SET Current = FALSE,
                ValidTo = v_now
            WHERE UnitID = p_unit_id AND Current = TRUE;

            -- Step 2: Insert the newly active revision
            INSERT INTO Unit (
                UnitVersionID,
                UnitID,
                Language,
                Curriculum,
                Grade,
                UnitName,
                UnitContent,
                ValidFrom,
                ValidTo,
                Current
            ) VALUES (
                UUID(),
                p_unit_id,
                p_language,
                p_curriculum,
                p_grade,
                p_unit_name,
                p_unit_content,
                v_now,
                NULL,
                TRUE
            );
        END;
        """,

        # 2. Update Lesson Version (SCD Type 2)
        "sp_update_lesson_scd2": """
        CREATE PROCEDURE sp_update_lesson_scd2(
            IN p_lesson_id VARCHAR(64),
            IN p_unit_id VARCHAR(64),
            IN p_lesson_name VARCHAR(255),
            IN p_lesson_content TEXT
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Expire current lesson version
            UPDATE Lesson
            SET Current = FALSE,
                ValidTo = v_now
            WHERE LessonID = p_lesson_id AND Current = TRUE;

            -- Step 2: Insert new active lesson version
            INSERT INTO Lesson (
                LessonVersionID,
                UnitID,
                LessonID,
                LessonName,
                LessonContent,
                ValidFrom,
                ValidTo,
                Current
            ) VALUES (
                UUID(),
                p_unit_id,
                p_lesson_id,
                p_lesson_name,
                p_lesson_content,
                v_now,
                NULL,
                TRUE
            );
        END;
        """,

        # 3. Update Learning Objective Version (SCD Type 2)
        "sp_update_learning_objective_scd2": """
        CREATE PROCEDURE sp_update_learning_objective_scd2(
            IN p_learning_objective_id VARCHAR(64),
            IN p_lesson_id VARCHAR(64),
            IN p_learning_objective_content TEXT
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Expire current objective version
            UPDATE LearningObjective
            SET Current = FALSE,
                ValidTo = v_now
            WHERE LearningObjectiveID = p_learning_objective_id AND Current = TRUE;

            -- Step 2: Insert new active objective version
            INSERT INTO LearningObjective (
                LearningObjectiveVersionID,
                LessonID,
                LearningObjectiveID,
                LearningObjectiveContent,
                ValidFrom,
                ValidTo,
                Current
            ) VALUES (
                UUID(),
                p_lesson_id,
                p_learning_objective_id,
                p_learning_objective_content,
                v_now,
                NULL,
                TRUE
            );
        END;
        """,

        # 4. Update Skill Cluster Version (SCD Type 2)
        "sp_update_skill_cluster_scd2": """
        CREATE PROCEDURE sp_update_skill_cluster_scd2(
            IN p_skill_cluster_id VARCHAR(64),
            IN p_lesson_id VARCHAR(64),
            IN p_unit_id VARCHAR(64),
            IN p_learning_objective_id VARCHAR(64),
            IN p_skill_cluster_content TEXT
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Expire current skill cluster version
            UPDATE SkillCluster
            SET Current = FALSE,
                ValidTo = v_now
            WHERE SkillClusterID = p_skill_cluster_id AND Current = TRUE;

            -- Step 2: Insert new active skill cluster version
            INSERT INTO SkillCluster (
                SkillClusterVersionID,
                SkillClusterID,
                LessonID,
                UnitID,
                LearningObjectiveID,
                SkillClusterContent,
                ValidFrom,
                ValidTo,
                Current
            ) VALUES (
                UUID(),
                p_skill_cluster_id,
                p_lesson_id,
                p_unit_id,
                p_learning_objective_id,
                p_skill_cluster_content,
                v_now,
                NULL,
                TRUE
            );
        END;
        """
    }

    try:
        with conn.cursor() as cursor:
            for proc_name, ddl in procedures.items():
                print(f"Creating procedure `{proc_name}`...")
                cursor.execute(f"DROP PROCEDURE IF EXISTS {proc_name};")
                cursor.execute(ddl)
        print("All SCD2 stored procedures created successfully.")
    except Exception as e:
        print(f"Failed to create stored procedures: {e}")
        raise
    finally:
        conn.close()

def create_student_scd_procedures():
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
        autocommit=True
    )

    procedures = {
        # -------------------------------------------------------------
        # 1. Stored Procedure for StudentMasteryLevels
        # -------------------------------------------------------------
        "sp_record_student_mastery": """
        CREATE PROCEDURE sp_record_student_mastery(
            IN p_student_id VARCHAR(64),
            IN p_skill_cluster_id VARCHAR(64),
            IN p_mastery FLOAT,
            IN p_content_gap BOOLEAN,
            IN p_content_gap_type VARCHAR(128)
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Deactivate any currently active mastery record for this skill
            UPDATE StudentMasteryLevels
            SET Current = FALSE
            WHERE StudentID = p_student_id
              AND SkillClusterID = p_skill_cluster_id
              AND Current = TRUE;

            -- Step 2: Insert the new active mastery snapshot
            INSERT INTO StudentMasteryLevels (
                StudentID,
                SkillClusterID,
                Mastery,
                ContentGap,
                ContentGapType,
                Timestamp,
                Current
            ) VALUES (
                p_student_id,
                p_skill_cluster_id,
                p_mastery,
                p_content_gap,
                p_content_gap_type,
                v_now,
                TRUE
            )
            ON DUPLICATE KEY UPDATE
                Mastery = VALUES(Mastery),
                ContentGap = VALUES(ContentGap),
                ContentGapType = VALUES(ContentGapType),
                Current = TRUE;
        END;
        """,

        # -------------------------------------------------------------
        # 2. Stored Procedure for StudentDiagnostics
        # -------------------------------------------------------------
        "sp_record_student_diagnostic": """
        CREATE PROCEDURE sp_record_student_diagnostic(
            IN p_student_id VARCHAR(64),
            IN p_diagnosis_id VARCHAR(64),
            IN p_delta FLOAT
        )
        BEGIN
            DECLARE v_now DATETIME;
            SET v_now = UTC_TIMESTAMP();

            -- Step 1: Deactivate the existing active diagnosis entry
            UPDATE StudentDiagnostics
            SET Current = FALSE
            WHERE StudentID = p_student_id
              AND DiagnosisID = p_diagnosis_id
              AND Current = TRUE;

            -- Step 2: Insert the new active diagnostic delta
            INSERT INTO StudentDiagnostics (
                StudentID,
                DiagnosisID,
                Delta,
                Timestamp,
                Current
            ) VALUES (
                p_student_id,
                p_diagnosis_id,
                p_delta,
                v_now,
                TRUE
            )
            ON DUPLICATE KEY UPDATE
                Delta = VALUES(Delta),
                Current = TRUE;
        END;
        """
    }

    try:
        with conn.cursor() as cursor:
            for proc_name, ddl in procedures.items():
                print(f"Creating procedure `{proc_name}`...")
                cursor.execute(f"DROP PROCEDURE IF EXISTS {proc_name};")
                cursor.execute(ddl)
        print("Both stored procedures created successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    create_scd2_stored_procedures()
    create_student_scd_procedures()