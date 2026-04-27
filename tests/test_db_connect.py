from db.db_connect import get_database_settings


def test_source_db_settings_fall_back_to_legacy_env(monkeypatch):
    monkeypatch.delenv("SOURCE_DB_USER", raising=False)
    monkeypatch.delenv("SOURCE_DB_PASSWORD", raising=False)
    monkeypatch.delenv("SOURCE_DB_HOST", raising=False)
    monkeypatch.delenv("SOURCE_DB_PORT", raising=False)
    monkeypatch.delenv("SOURCE_DB_NAME", raising=False)
    monkeypatch.setenv("DB_USER", "legacy_user")
    monkeypatch.setenv("DB_PASSWORD", "legacy_password")
    monkeypatch.setenv("DB_HOST", "legacy_host")
    monkeypatch.setenv("DB_PORT", "3307")
    monkeypatch.setenv("DB_NAME", "legacy_db")

    settings = get_database_settings("source")

    assert settings.user == "legacy_user"
    assert settings.password == "legacy_password"
    assert settings.host == "legacy_host"
    assert settings.port == "3307"
    assert settings.name == "legacy_db"


def test_datamart_db_settings_allow_independent_database(monkeypatch):
    monkeypatch.setenv("SOURCE_DB_USER", "src_user")
    monkeypatch.setenv("SOURCE_DB_PASSWORD", "src_password")
    monkeypatch.setenv("SOURCE_DB_HOST", "src_host")
    monkeypatch.setenv("SOURCE_DB_PORT", "3306")
    monkeypatch.setenv("SOURCE_DB_NAME", "source_db")
    monkeypatch.setenv("DATAMART_DB_NAME", "datamart_db")
    monkeypatch.delenv("DATAMART_DB_USER", raising=False)
    monkeypatch.delenv("DATAMART_DB_PASSWORD", raising=False)
    monkeypatch.delenv("DATAMART_DB_HOST", raising=False)
    monkeypatch.delenv("DATAMART_DB_PORT", raising=False)

    settings = get_database_settings("datamart")

    assert settings.user == "src_user"
    assert settings.password == "src_password"
    assert settings.host == "src_host"
    assert settings.port == "3306"
    assert settings.name == "datamart_db"


def test_source_db_settings_allow_blank_password(monkeypatch):
    monkeypatch.setenv("SOURCE_DB_USER", "src_user")
    monkeypatch.setenv("SOURCE_DB_PASSWORD", "")
    monkeypatch.setenv("SOURCE_DB_HOST", "src_host")
    monkeypatch.setenv("SOURCE_DB_PORT", "3306")
    monkeypatch.setenv("SOURCE_DB_NAME", "source_db")

    settings = get_database_settings("source")

    assert settings.password == ""
    assert settings.is_configured is True
    assert settings.url == "mysql+pymysql://src_user@src_host:3306/source_db"
