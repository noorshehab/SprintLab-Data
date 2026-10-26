import argparse

try:
    from .db_config import DBConfig, add_arguments, config_from_namespace
except ImportError:  # direct script execution
    from db_config import DBConfig, add_arguments, config_from_namespace


def run_migration(host: str | None = None, user: str | None = None,
                  password: str | None = None, port: int | str | None = None,
                  db_name: str | None = None):
    cfg = DBConfig.from_values(
        host=host, user=user, password=password, port=port, db_name=db_name)
    db_name = cfg.db_name

    print(f"Connecting to MySQL server at {cfg.host}:{cfg.port}...")
    conn = cfg.connect(include_database=False)

    table_queries = [
        # 1. Curriculum Hierarchy
        (
            "Unit",
            """
            CREATE TABLE IF NOT EXISTS `Unit` (
                `UnitVersionID` VARCHAR(64) PRIMARY KEY,
                `UnitID` VARCHAR(64) NOT NULL,
                `Language` VARCHAR(16) NOT NULL DEFAULT 'en',
                `Curriculum` VARCHAR(64) NOT NULL,
                `Grade` VARCHAR(32) NOT NULL,
                `UnitName` VARCHAR(255) NOT NULL,
                `UnitContent` TEXT,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "Lesson",
            """
            CREATE TABLE IF NOT EXISTS `Lesson` (
                `LessonVersionID` VARCHAR(64) PRIMARY KEY,
                `UnitID` VARCHAR(64) NOT NULL,
                `LessonID` VARCHAR(64) NOT NULL,
                `LessonName` VARCHAR(255) NOT NULL,
                `LessonContent` TEXT,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "LearningObjective",
            """
            CREATE TABLE IF NOT EXISTS `LearningObjective` (
                `LearningObjectiveVersionID` VARCHAR(64) PRIMARY KEY,
                `LessonID` VARCHAR(64) NOT NULL,
                `LearningObjectiveID` VARCHAR(64) NOT NULL,
                `LearningObjectiveContent` TEXT,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "SkillCluster",
            """
            CREATE TABLE IF NOT EXISTS `SkillCluster` (
                `SkillClusterVersionID` VARCHAR(64) PRIMARY KEY,
                `SkillClusterID` VARCHAR(64) NOT NULL,
                `LessonID` VARCHAR(64) NOT NULL,
                `UnitID` VARCHAR(64) NULL,
                `LearningObjectiveID` VARCHAR(64) NULL,
                `SkillClusterContent` TEXT,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE,
                INDEX `idx_skillcluster_logical` (`SkillClusterID`, `Current`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "skill_similarities",
            """
            CREATE TABLE IF NOT EXISTS `skill_similarities` (
                `skill_id` VARCHAR(64) NOT NULL,
                `similar_skill_id` VARCHAR(64) NOT NULL,
                PRIMARY KEY (`skill_id`, `similar_skill_id`),
                INDEX `idx_skill_similarities_skill` (`skill_id`),
                INDEX `idx_skill_similarities_similar` (`similar_skill_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),

        # 2. Questions, Options, Diagnostics & Prerequisites
        (
            "QuestionContent",
            """
            CREATE TABLE IF NOT EXISTS `QuestionContent` (
                `QuestionID` VARCHAR(64) PRIMARY KEY,
                `SkillClusterID` VARCHAR(64) NOT NULL,
                `QuestionText` TEXT NOT NULL,
                `QuestionType` VARCHAR(64) NOT NULL,
                `Current` BOOLEAN DEFAULT TRUE,
                INDEX `idx_questioncontent_skill` (`SkillClusterID`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "Options",
            """
            CREATE TABLE IF NOT EXISTS `Options` (
                `OptionID` VARCHAR(64) PRIMARY KEY,
                `QuestionID` VARCHAR(64) NOT NULL,
                `OptionText` TEXT NOT NULL,
                `IsCorrect` BOOLEAN NOT NULL DEFAULT FALSE,
                `DisplayOrder` INT DEFAULT 0,
                `ContentGapType` VARCHAR(128) NULL,
                `DistractorReadingTrap` VARCHAR(128) NULL,
                `DistractorImpulsive` BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (`QuestionID`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "QuestionDiagnostics",
            """
            CREATE TABLE IF NOT EXISTS `QuestionDiagnostics` (
                `DiagnosticsVersion` VARCHAR(64) PRIMARY KEY,
                `QuestionID` VARCHAR(64) NOT NULL,
                `SkillClusterID` VARCHAR(64) NULL,
                `BloomTax` VARCHAR(64) NULL,
                `TheoreticalTime` FLOAT NULL,
                `VariablesCount` INT DEFAULT 0,
                `LogicalSteps` INT DEFAULT 0,
                `TrickQuestion` BOOLEAN DEFAULT FALSE,
                `VisualDependancy` BOOLEAN DEFAULT FALSE,
                `RealWorldContext` BOOLEAN DEFAULT FALSE,
                `MultiConceptFlag` BOOLEAN DEFAULT FALSE,
                `CognitiveLoadIndex` INT NULL,
                `LanguageChallengeType` VARCHAR(128) NULL,
                `ReasoningLevel` VARCHAR(64) NULL,
                `LanguageLevel` VARCHAR(64) NULL,
                `DifficultyLevel` FLOAT DEFAULT 0.0,
                `p_g` FLOAT DEFAULT 0.0,
                `p_t` FLOAT DEFAULT 0.0,
                `p_s` FLOAT DEFAULT 0.0,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (`QuestionID`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "PrerequisiteConcepts",
            """
            CREATE TABLE IF NOT EXISTS `PrerequisiteConcepts` (
                `QuestionID` VARCHAR(64) NOT NULL,
                `SkillClusterID` VARCHAR(64) NOT NULL,
                PRIMARY KEY (`QuestionID`, `SkillClusterID`),
                FOREIGN KEY (`QuestionID`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE,
                INDEX `idx_prereq_skill` (`SkillClusterID`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),

        # 3. Match, Student, and Response Tables
        (
            "Match",
            """
            CREATE TABLE IF NOT EXISTS `Match` (
                `MatchID` VARCHAR(64) PRIMARY KEY,
                `CreatedTimeStamp` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `MatchType` VARCHAR(64) NOT NULL,
                `NumParticipants` INT DEFAULT 1
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "Student",
            """
            CREATE TABLE IF NOT EXISTS `Student` (
                `StudentID` VARCHAR(64) PRIMARY KEY,
                `GradeBand` VARCHAR(32) NOT NULL DEFAULT '',
                `Curriculum` VARCHAR(64) NOT NULL DEFAULT ''
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "Response",
            """
            CREATE TABLE IF NOT EXISTS `Response` (
                `MatchID` VARCHAR(64) NOT NULL,
                `StudentID` VARCHAR(64) NOT NULL,
                `QuestionID` VARCHAR(64) NOT NULL,
                `StressTriggers` INT DEFAULT 0,
                `OptionID` VARCHAR(64) NULL,
                `Error` BOOLEAN DEFAULT FALSE,
                `HintsUsed` INT DEFAULT 0,
                `TimeOut` BOOLEAN DEFAULT FALSE,
                `ResponseTime` FLOAT DEFAULT 0.0,
                PRIMARY KEY (`MatchID`, `StudentID`, `QuestionID`),
                FOREIGN KEY (`MatchID`) REFERENCES `Match`(`MatchID`) ON DELETE CASCADE,
                FOREIGN KEY (`StudentID`) REFERENCES `Student`(`StudentID`) ON DELETE CASCADE,
                FOREIGN KEY (`QuestionID`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE,
                FOREIGN KEY (`OptionID`) REFERENCES `Options`(`OptionID`) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),

        # 4. Student Class & Mastery Levels
        (
            "StudentClass",
            """
            CREATE TABLE IF NOT EXISTS `StudentClass` (
                `StudentID` VARCHAR(64) NOT NULL,
                `ClassID` VARCHAR(64) NOT NULL,
                PRIMARY KEY (`StudentID`, `ClassID`),
                FOREIGN KEY (`StudentID`) REFERENCES `Student`(`StudentID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "StudentMasteryLevels",
            """
            CREATE TABLE IF NOT EXISTS `StudentMasteryLevels` (
                `StudentID` VARCHAR(64) NOT NULL,
                `SkillClusterID` VARCHAR(64) NOT NULL,
                `Mastery` FLOAT DEFAULT 0.0,
                `ContentGap` BOOLEAN DEFAULT FALSE,
                `ContentGapType` VARCHAR(128) NULL,
                `Timestamp` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `Current` BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (`StudentID`, `SkillClusterID`, `Timestamp`),
                FOREIGN KEY (`StudentID`) REFERENCES `Student`(`StudentID`) ON DELETE CASCADE,
                INDEX `idx_mastery_skill` (`SkillClusterID`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),

        # 5. Diagnostic Thresholds, Student Diagnostics & Treatments
        (
            "DiagnosticThresholds",
            """
            CREATE TABLE IF NOT EXISTS `DiagnosticThresholds` (
                `DiagnosisVersionID` VARCHAR(64) PRIMARY KEY,
                `DiagnosisID` VARCHAR(64) NOT NULL,
                `DiagnosisName` VARCHAR(128) NOT NULL,
                `Threshold` FLOAT NOT NULL,
                `ValidFrom` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `ValidTo` DATETIME NULL,
                `Current` BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "StudentDiagnostics",
            """
            CREATE TABLE IF NOT EXISTS `StudentDiagnostics` (
                `StudentID` VARCHAR(64) NOT NULL,
                `DiagnosisID` VARCHAR(64) NOT NULL,
                `Delta` FLOAT DEFAULT 0.0,
                `Timestamp` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                `Current` BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (`StudentID`, `DiagnosisID`, `Timestamp`),
                FOREIGN KEY (`StudentID`) REFERENCES `Student`(`StudentID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "Treatment",
            """
            CREATE TABLE IF NOT EXISTS `Treatment` (
                `StudentID` VARCHAR(64) NOT NULL,
                `DiagnosisID` VARCHAR(64) NOT NULL,
                `QuestionAttribute` VARCHAR(64) NOT NULL,
                `Value` VARCHAR(128) NOT NULL,
                `QueryOperator` VARCHAR(16) NOT NULL DEFAULT '=',
                PRIMARY KEY (`StudentID`, `DiagnosisID`, `QuestionAttribute`),
                FOREIGN KEY (`StudentID`) REFERENCES `Student`(`StudentID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),

        # 6. Hints Tables
        (
            "pre_hints",
            """
            CREATE TABLE IF NOT EXISTS `pre_hints` (
                `hint_id` VARCHAR(64) PRIMARY KEY,
                `hint_text` TEXT NOT NULL,
                `topic` VARCHAR(128) NULL,
                `content_gap_type` VARCHAR(128) NULL,
                `diagnosis` VARCHAR(128) NULL,
                `hint_type` VARCHAR(64) DEFAULT 'content_gap'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "post_hints",
            """
            CREATE TABLE IF NOT EXISTS `post_hints` (
                `hint_id` VARCHAR(64) PRIMARY KEY,
                `hint_text` TEXT NOT NULL,
                `question_id` VARCHAR(64) NOT NULL,
                `hint_order` INT DEFAULT 0,
                FOREIGN KEY (`question_id`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
        (
            "play_hints",
            """
            CREATE TABLE IF NOT EXISTS `play_hints` (
                `hint_id` VARCHAR(64) PRIMARY KEY,
                `hint_text` TEXT NOT NULL,
                `q_id` VARCHAR(64) NOT NULL,
                `hint_order` INT DEFAULT 0,
                FOREIGN KEY (`q_id`) REFERENCES `QuestionContent`(`QuestionID`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        ),
    ]

    try:
        with conn.cursor() as cursor:
            print(f"Ensuring database `{db_name}` exists...")
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            cursor.execute(f"USE `{db_name}`;")

            for table_name, query in table_queries:
                print(f"Creating table `{table_name}`...")
                cursor.execute(query)

            cursor.execute("SHOW TABLES;")
            existing_tables = [list(row.values())[0] for row in cursor.fetchall()]
            print(f"\nSuccessfully verified {len(existing_tables)} tables in `{db_name}`.")
    finally:
        conn.close()

def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(description="Create SprintLab MySQL tables"))
    args = parser.parse_args(argv)
    run_migration(**vars(config_from_namespace(args)))


if __name__ == "__main__":
    main()