from unittest.mock import MagicMock


def test_dash_app_creates_server():
    from dashboard.app import create_dash_app

    mock_db = MagicMock()
    mock_db.conn = MagicMock()
    app = create_dash_app(db=mock_db)
    assert app is not None
    assert app.server is not None
