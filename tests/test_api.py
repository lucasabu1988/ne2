from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

def test_health_endpoint():
    with patch("app.initialize_components") as mock_init:
        mock_init.return_value = {
            "db": MagicMock(), "ingestion": MagicMock(),
            "prediction_engine": MagicMock(), "trading_engine": MagicMock(),
            "risk_manager": MagicMock(), "portfolio": MagicMock(),
        }
        from app import create_api
        api = create_api(mock_init.return_value)
        client = TestClient(api)
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
