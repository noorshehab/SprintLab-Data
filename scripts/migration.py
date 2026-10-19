import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def run_migration():
    host = os.getenv("DB_HOST", "localhost")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "password")
    port = int(os.getenv("DB_PORT", "3306"))
    db_name = os.getenv("DB_NAME", "sprintlab_db")

    print(f"Connecting to MySQL server at {host}:{port}...")
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )

    table_queries = [
        # =====================================================================
        # 1. Curriculum Hierarchy (With SCD2 fields matching the diagram)
        # =====================================================================
        """
        CREATE TABLE IF NOT EXISTS Unit (
            UnitVersionID VARCHAR(64) PRIMARY KEY,
            UnitID VARCHAR(64) NOT NULL,
            Language VARCHAR(16) NOT NULL DEFAULT 'en',
            Curriculum VARCHAR(64) NOT NULL,
            Grade VARCHAR(32) NOT NULL,
            UnitName VARCHAR(255) NOT NULL,
            UnitContent TEXT,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            INDEX idx_unit_unitid (UnitID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Lesson (
            LessonVersionID VARCHAR(64) PRIMARY KEY,
            UnitID VARCHAR(64) NOT NULL,
            LessonID VARCHAR(64) NOT NULL,
            LessonName VARCHAR(255) NOT NULL,
            LessonContent TEXT,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            INDEX idx_lesson_lessonid (LessonID),
            INDEX idx_lesson_unitid (UnitID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS LearningObjective (
            LearningObjectiveVersionID VARCHAR(64) PRIMARY KEY,
            LessonID VARCHAR(64) NOT NULL,
            LearningObjectiveID VARCHAR(64) NOT NULL,
            LearningObjectiveContent TEXT,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            INDEX idx_learningobjective_id (LearningObjectiveID),
            INDEX idx_learningobjective_lessonid (LessonID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS SkillCluster (
            SkillClusterID VARCHAR(64) PRIMARY KEY,
            LessonID VARCHAR(64) NOT NULL,
            UnitID VARCHAR(64),
            LearningObjectiveID VARCHAR(64),
            SkillClusterContent TEXT,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            INDEX idx_skillcluster_lessonid (LessonID),
            INDEX idx_skillcluster_unitid (UnitID),
            INDEX idx_skillcluster_objectiveid (LearningObjectiveID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # =====================================================================
        # 2. Questions, Options & Diagnostics
        # =====================================================================
        """
        CREATE TABLE IF NOT EXISTS QuestionContent (
            QuestionID VARCHAR(64) PRIMARY KEY,
            SkillClusterID VARCHAR(64) NOT NULL,
            QuestionText TEXT NOT NULL,
            QuestionType VARCHAR(64) NOT NULL,
            Current BOOLEAN DEFAULT TRUE,
            FOREIGN KEY (SkillClusterID) REFERENCES SkillCluster(SkillClusterID) ON DELETE CASCADE,
            INDEX idx_questioncontent_skillcluster (SkillClusterID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Options (
            OptionID VARCHAR(64) PRIMARY KEY,
            QuestionID VARCHAR(64) NOT NULL,
            OptionText TEXT NOT NULL,
            IsCorrect BOOLEAN NOT NULL DEFAULT FALSE,
            DisplayOrder INT DEFAULT 0,
            ContentGapType VARCHAR(128),
            DistractorReadingType VARCHAR(128),
            DistractorImpulsive BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (QuestionID) REFERENCES QuestionContent(QuestionID) ON DELETE CASCADE,
            INDEX idx_options_questionid (QuestionID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS QuestionDiagnostics (
            DiagnosticsVersion VARCHAR(64) PRIMARY KEY,
            QuestionID VARCHAR(64) NOT NULL,
            BloomTax VARCHAR(64),
            TheoreticalTime FLOAT,
            VariablesCount INT DEFAULT 0,
            LogicalSteps INT DEFAULT 0,
            TrickQuestion BOOLEAN DEFAULT FALSE,
            VisualDependancy BOOLEAN DEFAULT FALSE,
            RealWorldContext BOOLEAN DEFAULT FALSE,
            MultiConceptFlag BOOLEAN DEFAULT FALSE,
            CognitiveLoadIndex INT,
            LanguageChallengeType VARCHAR(128),
            ReasoningLevel VARCHAR(64),
            LanguageLevel VARCHAR(64),
            DifficultyLevel FLOAT DEFAULT 0.0,
            p_g FLOAT DEFAULT 0.0,
            p_t FLOAT DEFAULT 0.0,
            p_s FLOAT DEFAULT 0.0,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            FOREIGN KEY (QuestionID) REFERENCES QuestionContent(QuestionID) ON DELETE CASCADE,
            INDEX idx_qdiag_questionid (QuestionID),
            INDEX idx_qdiag_treatment_search (BloomTax, DifficultyLevel, CognitiveLoadIndex)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS PrerequisiteConcepts (
            QuestionID VARCHAR(64) NOT NULL,
            SkillClusterID VARCHAR(64) NOT NULL,
            PRIMARY KEY (QuestionID, SkillClusterID),
            FOREIGN KEY (QuestionID) REFERENCES QuestionContent(QuestionID) ON DELETE CASCADE,
            FOREIGN KEY (SkillClusterID) REFERENCES SkillCluster(SkillClusterID) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # =====================================================================
        # 3. Match, Students, and Responses
        # =====================================================================
        """
        CREATE TABLE IF NOT EXISTS MatchTable (
            MatchID VARCHAR(64) PRIMARY KEY,
            CreatedTimeStamp DATETIME NOT NULL,
            MatchType VARCHAR(64) NOT NULL,
            NumParticipants INT DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Student (
            StudentID VARCHAR(64) PRIMARY KEY,
            GradeBand VARCHAR(32) NOT NULL,
            Curriculum VARCHAR(64) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Response (
            MatchID VARCHAR(64) NOT NULL,
            StudentID VARCHAR(64) NOT NULL,
            QuestionID VARCHAR(64) NOT NULL,
            StressTriggers INT DEFAULT 0,
            OptionID VARCHAR(64),
            Error BOOLEAN DEFAULT FALSE,
            HintsUsed INT DEFAULT 0,
            TimeOut BOOLEAN DEFAULT FALSE,
            ResponseTime FLOAT DEFAULT 0.0,
            PRIMARY KEY (MatchID, StudentID, QuestionID),
            FOREIGN KEY (MatchID) REFERENCES MatchTable(MatchID) ON DELETE CASCADE,
            FOREIGN KEY (StudentID) REFERENCES Student(StudentID) ON DELETE CASCADE,
            FOREIGN KEY (QuestionID) REFERENCES QuestionContent(QuestionID) ON DELETE CASCADE,
            FOREIGN KEY (OptionID) REFERENCES Options(OptionID) ON DELETE SET NULL,
            INDEX idx_response_match_student (MatchID, StudentID),
            INDEX idx_response_question (QuestionID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # =====================================================================
        # 4. Student Metadata & Mastery Levels
        # =====================================================================
        """
        CREATE TABLE IF NOT EXISTS StudentClass (
            StudentID VARCHAR(64) NOT NULL,
            ClassID VARCHAR(64) NOT NULL,
            PRIMARY KEY (StudentID, ClassID),
            FOREIGN KEY (StudentID) REFERENCES Student(StudentID) ON DELETE CASCADE,
            INDEX idx_studentclass_classid (ClassID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS StudentMasteryLevels (
            StudentID VARCHAR(64) NOT NULL,
            SkillClusterID VARCHAR(64) NOT NULL,
            Mastery FLOAT DEFAULT 0.0,
            ContentGap BOOLEAN DEFAULT FALSE,
            ContentGapType VARCHAR(128),
            Timestamp DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            PRIMARY KEY (StudentID, SkillClusterID),
            FOREIGN KEY (StudentID) REFERENCES Student(StudentID) ON DELETE CASCADE,
            FOREIGN KEY (SkillClusterID) REFERENCES SkillCluster(SkillClusterID) ON DELETE CASCADE,
            INDEX idx_masterylevels_skillcluster (SkillClusterID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # =====================================================================
        # 5. Diagnostic Thresholds, Diagnostics & Treatments
        # =====================================================================
        """
        CREATE TABLE IF NOT EXISTS DiagnosticThresholds (
            DiagnosisVersionID VARCHAR(64) PRIMARY KEY,
            DiagnosisID VARCHAR(64) NOT NULL,
            DiagnosisName VARCHAR(128) NOT NULL,
            Threshold FLOAT NOT NULL,
            ValidFrom DATETIME,
            ValidTo DATETIME,
            Current BOOLEAN DEFAULT TRUE,
            INDEX idx_diagthresh_diagnosisid (DiagnosisID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS StudentDiagnostics (
            StudentID VARCHAR(64) NOT NULL,
            DiagnosisID VARCHAR(64) NOT NULL,
            Delta FLOAT DEFAULT 0.0,
            PRIMARY KEY (StudentID, DiagnosisID),
            FOREIGN KEY (StudentID) REFERENCES Student(StudentID) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Treatment (
            StudentID VARCHAR(64) NOT NULL,
            DiagnosisID VARCHAR(64) NOT NULL,
            QuestionAttribute VARCHAR(64) NOT NULL,
            Value VARCHAR(128) NOT NULL,
            QueryOperator VARCHAR(16) NOT NULL DEFAULT '=',
            PRIMARY KEY (StudentID, DiagnosisID, QuestionAttribute),
            FOREIGN KEY (StudentID) REFERENCES Student(StudentID) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    ]

    try:
        with conn.cursor() as cursor:
            print(f"Ensuring database `{db_name}` exists...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            cursor.execute(f"USE `{db_name}`;")

            for query in table_queries:
                cursor.execute(query)

        conn.commit()
        print(f"All tables and indexes matching the schema created successfully in database `{db_name}`.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to create schema: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()