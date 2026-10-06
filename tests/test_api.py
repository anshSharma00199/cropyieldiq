"""End-to-end API tests (FastAPI TestClient, in-memory SQLite). Needs: pip install -r requirements-dev.txt"""

import dataclasses
import itertools
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.database import SessionLocal
from app.main import app
from app.models import User, YieldOutcome, YieldPrediction

PASSWORD = "Str0ng-Passw0rd"
_n = itertools.count()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs the lifespan: creates tables and loads the model
        yield c


def fresh_ip():
    return f"10.0.{next(_n) // 250}.{next(_n) % 250}"


def signup(client, role=None):
    """Create a user (optionally promoted via DB) and return (auth headers, user email)."""
    email = f"user{next(_n)}@example.com"
    ip = {"X-Real-IP": fresh_ip()}
    assert client.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD}, headers=ip).status_code == 201
    if role:
        with SessionLocal() as db:
            u = db.query(User).filter_by(email=email).one()
            u.role = role
            db.commit()
    tok = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD}, headers=ip).json()
    return {"Authorization": f"Bearer {tok['access_token']}"}, email, tok


def crop_inputs(crop):
    from tests.test_ml import profile as p

    return p(crop)


def test_health(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["model"] and r.json()["database"]


def test_meta_is_public_and_cacheable(client):
    r = client.get("/api/v1/meta")
    assert r.status_code == 200 and "public" in r.headers["cache-control"]
    assert len(r.json()["crops"]) == 22


def test_security_headers_and_request_id(client):
    r = client.get("/health/live", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123" and r.headers["x-content-type-options"] == "nosniff"


def test_predict_requires_authentication(client):
    assert client.post("/api/v1/predict", json=crop_inputs("rice")).status_code == 401


def test_register_validation(client):
    ip = {"X-Real-IP": fresh_ip()}
    assert client.post("/api/v1/auth/register", json={"email": "a@b.co", "password": "short1"}, headers=ip).status_code == 422
    assert client.post("/api/v1/auth/register", json={"email": "not-an-email", "password": PASSWORD}, headers=ip).status_code == 422
    _, email, _ = signup(client)
    dup = client.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD}, headers=ip)
    assert dup.status_code == 409


def test_login_failure_and_account_lockout(client):
    _, email, _ = signup(client)
    ip = {"X-Real-IP": fresh_ip()}
    codes = [
        client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password-1"}, headers=ip).status_code for _ in range(6)
    ]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    # even the right password is refused while locked
    assert client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD}, headers=ip).status_code == 429


def test_predict_flow_history_feedback(client):
    h, _, _ = signup(client)
    r = client.post("/api/v1/predict", json=crop_inputs("rice"), headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"] == "rice" and body["top3"][0]["probability"] > 0.5 and body["model_version"]
    hist = client.get("/api/v1/history", headers=h).json()
    assert hist[0]["id"] == body["id"] and hist[0]["crop"] == "rice"
    assert client.post("/api/v1/feedback", json={"prediction_id": body["id"], "helpful": True}, headers=h).status_code == 201
    other, _, _ = signup(client)  # users cannot rate (or see) each other's predictions
    assert client.post("/api/v1/feedback", json={"prediction_id": body["id"], "helpful": False}, headers=other).status_code == 404
    assert client.get("/api/v1/history", headers=other).json() == []


def test_different_inputs_give_different_outputs(client):
    h, _, _ = signup(client)
    results = {
        c: client.post("/api/v1/predict", json=crop_inputs(c), headers=h).json()["recommendation"]
        for c in ("rice", "coffee", "chickpea", "mango", "apple")
    }
    assert len(set(results.values())) == 5


def test_input_validation(client):
    h, _, _ = signup(client)
    bad = dict(crop_inputs("rice"), ph=20)
    assert client.post("/api/v1/predict", json=bad, headers=h).status_code == 422
    assert client.post("/api/v1/predict", json={"N": 1}, headers=h).status_code == 422


def test_sensitivity_endpoint(client):
    h, _, _ = signup(client)
    r = client.post("/api/v1/predict/sensitivity", json={"inputs": crop_inputs("rice"), "feature": "rainfall", "points": 8}, headers=h)
    assert r.status_code == 200 and len(r.json()["curve"]) == 8


def test_rbac_matrix(client):
    farmer, _, _ = signup(client)
    expert, _, _ = signup(client, "expert")
    admin, _, _ = signup(client, "admin")
    assert client.get("/api/v1/admin/stats", headers=farmer).status_code == 403
    assert client.get("/api/v1/admin/stats", headers=expert).status_code == 200
    assert client.get("/api/v1/admin/users", headers=expert).status_code == 403
    assert client.get("/api/v1/admin/users", headers=admin).status_code == 200
    assert client.get("/api/v1/admin/audit", headers=admin).status_code == 200
    assert client.get("/api/v1/admin/stats").status_code == 401


def test_admin_can_change_role_and_it_is_audited(client):
    admin, _, _ = signup(client, "admin")
    _, email, _ = signup(client)
    uid = next(u["id"] for u in client.get("/api/v1/admin/users?limit=200", headers=admin).json() if u["email"] == email)
    assert client.patch(f"/api/v1/admin/users/{uid}/role", json={"role": "expert"}, headers=admin).json()["role"] == "expert"
    assert any(a["action"] == "role_changed" for a in client.get("/api/v1/admin/audit", headers=admin).json())
    assert client.patch(f"/api/v1/admin/users/{uid}/role", json={"role": "superuser"}, headers=admin).status_code == 422


def test_logout_revokes_access_token(client):
    h, _, tok = signup(client)
    assert client.get("/api/v1/auth/me", headers=h).status_code == 200
    assert client.post("/api/v1/auth/logout", headers=h).status_code == 204
    assert client.get("/api/v1/auth/me", headers=h).status_code == 401


def test_refresh_token_rotation(client):
    _, _, tok = signup(client)
    ip = {"X-Real-IP": fresh_ip()}
    first = client.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]}, headers=ip)
    assert first.status_code == 200
    again = client.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]}, headers=ip)
    assert again.status_code == 401  # a refresh token works only once
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": tok["access_token"]}, headers=ip).status_code == 401


def test_auth_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(deps, "settings", dataclasses.replace(deps.settings, rl_auth=(3, 60)))
    ip = {"X-Real-IP": fresh_ip()}
    codes = [client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x" * 12}, headers=ip) for _ in range(4)]
    assert [c.status_code for c in codes] == [401, 401, 401, 429]
    assert int(codes[3].headers["retry-after"]) > 0


def test_request_body_size_limit(client):
    r = client.post(
        "/api/v1/auth/login", content=b"{" + b"a" * 70000 + b"}", headers={"Content-Type": "application/json", "X-Real-IP": fresh_ip()}
    )
    assert r.status_code == 413


class FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def fake_upstream(temp, hum, rain):
    def _get(url, params=None, timeout=None):
        if "geocoding" in url:
            return FakeResp(
                {
                    "results": [
                        {
                            "latitude": (24.25 if (params or {}).get("name") == "DryTown" else 23.25),
                            "longitude": (78.40 if (params or {}).get("name") == "DryTown" else 77.40),
                            "name": params["name"],
                            "admin1": "Madhya Pradesh",
                        }
                    ]
                }
            )
        n = 24 * 31
        return FakeResp(
            {"hourly": {"temperature_2m": [temp] * n, "relative_humidity_2m": [hum] * n}, "daily": {"precipitation_sum": [rain / 31] * 31}}
        )

    return _get


def test_live_prediction_uses_fetched_weather_and_changes_with_it(client):
    h, _, _ = signup(client)
    soil = {"N": 90, "P": 42, "K": 43, "ph": 6.5}
    with patch("app.services.weather.requests.get", side_effect=fake_upstream(24, 85, 230)):
        wet = client.post("/api/v1/predict/live", json={"district": "WetTown", **soil}, headers=h)
    with patch("app.services.weather.requests.get", side_effect=fake_upstream(30, 30, 25)):
        dry = client.post("/api/v1/predict/live", json={"district": "DryTown", **soil}, headers=h)
    assert wet.status_code == 200 and dry.status_code == 200
    assert wet.json()["weather"]["rainfall"] != dry.json()["weather"]["rainfall"]
    assert wet.json()["recommendation"] != dry.json()["recommendation"]
    assert wet.json()["assumptions"]


def test_live_prediction_returns_503_when_weather_is_down(client):
    h, _, _ = signup(client)
    with patch("app.services.weather.requests.get", side_effect=ConnectionError("boom")):
        r = client.post("/api/v1/predict/live", json={"district": "NowhereVille", "N": 90, "P": 42, "K": 43, "ph": 6.5}, headers=h)
    assert r.status_code == 503  # never a made-up default
    assert client.post("/api/v1/predict/live", json={"N": 1, "P": 1, "K": 1, "ph": 6}, headers=h).status_code == 422


def test_yield_prediction_returns_warning_for_out_of_range_area(client):
    h, _, _ = signup(client)

    payload = {
        "state": "Andhra Pradesh",
        "crop": "rice",
        "area_ha": 100,
        "year": 2017,
        "lag_yield": 1372,
        "temp_mean": 27.8,
        "rainfall": 926.1,
    }

    response = client.post(
        "/api/v1/yield/predict",
        json=payload,
        headers=h,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["yield_kg_per_ha"] > 0
    assert result["interval_lo"] <= result["yield_kg_per_ha"]
    assert result["yield_kg_per_ha"] <= result["interval_hi"]
    assert result["model_version"]
    assert any("Area" in warning and "central 98%" in warning for warning in result["warnings"])
    assert result["disclaimer"]


def test_yield_prediction_has_no_area_warning_for_typical_area(client):
    h, _, _ = signup(client)

    payload = {
        "state": "Andhra Pradesh",
        "crop": "rice",
        "area_ha": 1000,
        "year": 2017,
        "lag_yield": 1372,
        "temp_mean": 27.8,
        "rainfall": 926.1,
    }

    response = client.post(
        "/api/v1/yield/predict",
        json=payload,
        headers=h,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["yield_kg_per_ha"] > 0
    assert not any("Area" in warning for warning in result["warnings"])


def test_yield_prediction_requires_authentication(client):
    payload = {
        "state": "Andhra Pradesh",
        "crop": "rice",
        "area_ha": 1000,
        "year": 2017,
    }

    response = client.post("/api/v1/yield/predict", json=payload)

    assert response.status_code == 401


def test_crop_planner_returns_crop_specific_guidance(client):
    h, _, _ = signup(client)
    r = client.post(
        "/api/v1/planner",
        json={"crop": "rice", "inputs": crop_inputs("rice")},
        headers=h,
    )

    assert r.status_code == 200
    body = r.json()
    assert body["crop"] == "rice"
    assert body["typical_ranges"]
    assert body["feature_status"]
    assert body["advice"]
    assert body["disclaimer"]


def test_crop_planner_rejects_unknown_crop(client):
    h, _, _ = signup(client)
    r = client.post(
        "/api/v1/planner",
        json={"crop": "unknown_crop", "inputs": crop_inputs("rice")},
        headers=h,
    )

    assert r.status_code == 404


def test_crop_planner_requires_authentication(client):
    r = client.post(
        "/api/v1/planner",
        json={"crop": "rice", "inputs": crop_inputs("rice")},
    )

    assert r.status_code == 401


def test_yield_prediction_is_saved_with_prediction_id(client):
    h, _, _ = signup(client)
    payload = {
        "state": "Andhra Pradesh",
        "crop": "rice",
        "area_ha": 1000,
        "year": 2017,
        "lag_yield": 1372,
        "temp_mean": 27.8,
        "rainfall": 926.1,
    }
    r = client.post("/api/v1/yield/predict", json=payload, headers=h)
    assert r.status_code == 200
    data = r.json()
    assert "prediction_id" in data
    assert isinstance(data["prediction_id"], int)
    assert data["prediction_id"] > 0

    with SessionLocal() as db:
        saved = db.get(YieldPrediction, data["prediction_id"])
        assert saved is not None
        assert saved.yield_kg_per_ha == data["yield_kg_per_ha"]
        assert saved.interval_lo == data["interval_lo"]
        assert saved.interval_hi == data["interval_hi"]
        assert saved.model_version == data["model_version"]
        assert saved.inputs["crop"] == "rice"


def test_user_can_submit_actual_yield_for_own_prediction(client):
    h, _, _ = signup(client)
    pred_res = client.post(
        "/api/v1/yield/predict",
        json={"state": "Andhra Pradesh", "crop": "rice", "area_ha": 1000, "year": 2017},
        headers=h,
    )
    assert pred_res.status_code == 200, pred_res.text
    pred_id = pred_res.json()["prediction_id"]

    outcome_res = client.post(
        "/api/v1/yield/outcome",
        json={
            "yield_prediction_id": pred_id,
            "actual_yield_kg_per_ha": 2450.5,
            "harvest_date": "2026-10-01",
            "notes": "Good monsoon harvest",
        },
        headers=h,
    )
    assert outcome_res.status_code == 201
    out_data = outcome_res.json()
    assert out_data["status"] == "recorded"
    assert out_data["id"] > 0
    assert out_data["yield_prediction_id"] == pred_id
    assert out_data["actual_yield_kg_per_ha"] == 2450.5

    with SessionLocal() as db:
        outcome_row = db.get(YieldOutcome, out_data["id"])
        assert outcome_row is not None
        assert outcome_row.actual_yield_kg_per_ha == 2450.5
        assert str(outcome_row.harvest_date) == "2026-10-01"
        assert outcome_row.notes == "Good monsoon harvest"


def test_user_cannot_submit_outcome_for_another_users_prediction(client):
    user_a, _, _ = signup(client)
    user_b, _, _ = signup(client)

    pred_res = client.post(
        "/api/v1/yield/predict",
        json={"state": "Andhra Pradesh", "crop": "rice", "area_ha": 1000, "year": 2017},
        headers=user_a,
    )
    pred_id = pred_res.json()["prediction_id"]

    res = client.post(
        "/api/v1/yield/outcome",
        json={"yield_prediction_id": pred_id, "actual_yield_kg_per_ha": 2500.0},
        headers=user_b,
    )
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_yield_outcome_unknown_prediction_returns_404(client):
    h, _, _ = signup(client)
    res = client.post(
        "/api/v1/yield/outcome",
        json={"yield_prediction_id": 999999, "actual_yield_kg_per_ha": 2000.0},
        headers=h,
    )
    assert res.status_code == 404


def test_duplicate_outcome_returns_409(client):
    h, _, _ = signup(client)
    pred_res = client.post(
        "/api/v1/yield/predict",
        json={"state": "Andhra Pradesh", "crop": "rice", "area_ha": 1000, "year": 2017},
        headers=h,
    )
    pred_id = pred_res.json()["prediction_id"]

    first = client.post(
        "/api/v1/yield/outcome",
        json={"yield_prediction_id": pred_id, "actual_yield_kg_per_ha": 2100.0},
        headers=h,
    )
    assert first.status_code == 201

    second = client.post(
        "/api/v1/yield/outcome",
        json={"yield_prediction_id": pred_id, "actual_yield_kg_per_ha": 2200.0},
        headers=h,
    )
    assert second.status_code == 409


def test_yield_outcome_validation_rejects_invalid_values(client):
    h, _, _ = signup(client)
    pred_res = client.post(
        "/api/v1/yield/predict",
        json={"state": "Andhra Pradesh", "crop": "rice", "area_ha": 1000, "year": 2017},
        headers=h,
    )
    pred_id = pred_res.json()["prediction_id"]

    # Zero actual yield
    assert (
        client.post(
            "/api/v1/yield/outcome",
            json={
                "yield_prediction_id": pred_id,
                "actual_yield_kg_per_ha": 0,
            },
            headers=h,
        ).status_code
        == 422
    )

    # Negative actual yield
    assert (
        client.post(
            "/api/v1/yield/outcome",
            json={"yield_prediction_id": pred_id, "actual_yield_kg_per_ha": -100.0},
            headers=h,
        ).status_code
        == 422
    )

    # Excessive actual yield (> 100,000)
    assert (
        client.post(
            "/api/v1/yield/outcome",
            json={"yield_prediction_id": pred_id, "actual_yield_kg_per_ha": 150000.0},
            headers=h,
        ).status_code
        == 422
    )

    # Invalid prediction_id (0)
    assert (
        client.post(
            "/api/v1/yield/outcome",
            json={"yield_prediction_id": 0, "actual_yield_kg_per_ha": 1000.0},
            headers=h,
        ).status_code
        == 422
    )


def test_yield_evaluation_rbac(client):
    farmer, _, _ = signup(client)
    expert, _, _ = signup(client, "expert")
    admin, _, _ = signup(client, "admin")

    assert client.get("/api/v1/admin/yield/evaluation").status_code == 401
    assert client.get("/api/v1/admin/yield/evaluation", headers=farmer).status_code == 403
    assert client.get("/api/v1/admin/yield/evaluation", headers=expert).status_code == 200
    assert client.get("/api/v1/admin/yield/evaluation", headers=admin).status_code == 200


def test_yield_evaluation_empty_and_known_metrics(client):
    admin, _, _ = signup(client, "admin")

    # Step 1: Clean table for empty evaluation test
    with SessionLocal() as db:
        db.query(YieldOutcome).delete()
        db.query(YieldPrediction).delete()
        db.commit()

    empty_res = client.get("/api/v1/admin/yield/evaluation", headers=admin)
    assert empty_res.status_code == 200
    empty_data = empty_res.json()
    assert empty_data["sample_count"] == 0
    assert empty_data["count"] == 0
    assert empty_data["mae"] is None
    assert empty_data["rmse"] is None
    assert empty_data["mean_error"] is None
    assert empty_data["bias"] is None
    assert empty_data["mape"] is None
    assert empty_data["inside_interval_count"] == 0
    assert empty_data["coverage_percentage"] is None
    assert empty_data["disclaimer"]

    # Step 2: Insert small known example
    # Row 1: pred = 1100.0, lo = 900.0, hi = 1300.0, actual = 1000.0 (err = +100, abs = 100, sq = 10000, inside = True)
    # Row 2: pred = 1900.0, lo = 1700.0, hi = 2100.0, actual = 2000.0 (err = -100, abs = 100, sq = 10000, inside = True)
    with SessionLocal() as db:
        admin_user = db.query(User).filter_by(role="admin").first()
        p1 = YieldPrediction(
            user_id=admin_user.id,
            inputs={"crop": "rice"},
            yield_kg_per_ha=1100.0,
            interval_lo=900.0,
            interval_hi=1300.0,
            model_version="test-v1",
        )
        p2 = YieldPrediction(
            user_id=admin_user.id,
            inputs={"crop": "maize"},
            yield_kg_per_ha=1900.0,
            interval_lo=1700.0,
            interval_hi=2100.0,
            model_version="test-v1",
        )
        db.add_all([p1, p2])
        db.commit()

        o1 = YieldOutcome(
            yield_prediction_id=p1.id,
            user_id=admin_user.id,
            actual_yield_kg_per_ha=1000.0,
            notes="test 1",
        )
        o2 = YieldOutcome(
            yield_prediction_id=p2.id,
            user_id=admin_user.id,
            actual_yield_kg_per_ha=2000.0,
            notes="test 2",
        )
        db.add_all([o1, o2])
        db.commit()

    res = client.get("/api/v1/admin/yield/evaluation", headers=admin)
    assert res.status_code == 200
    data = res.json()
    assert data["sample_count"] == 2
    assert data["count"] == 2
    assert data["mae"] == 100.0
    assert data["rmse"] == 100.0
    assert data["mean_error"] == 0.0
    assert data["bias"] == 0.0
    assert data["mape"] == 7.5
    assert data["inside_interval_count"] == 2
    assert data["coverage_percentage"] == 100.0

    # Step 3: Add Row 3 that falls outside interval to verify coverage calculation
    # Row 3: pred = 3000.0, lo = 2800.0, hi = 3200.0, actual = 3500.0
    # err = 3000 - 3500 = -500, abs = 500, sq = 250000, inside = False
    with SessionLocal() as db:
        admin_user = db.query(User).filter_by(role="admin").first()
        p3 = YieldPrediction(
            user_id=admin_user.id,
            inputs={"crop": "chickpea"},
            yield_kg_per_ha=3000.0,
            interval_lo=2800.0,
            interval_hi=3200.0,
            model_version="test-v1",
        )
        db.add(p3)
        db.commit()

        o3 = YieldOutcome(
            yield_prediction_id=p3.id,
            user_id=admin_user.id,
            actual_yield_kg_per_ha=3500.0,
            notes="test 3 outside",
        )
        db.add(o3)
        db.commit()

    res3 = client.get("/api/v1/admin/yield/evaluation", headers=admin)
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["sample_count"] == 3
    assert data3["count"] == 3
    # MAE = (100 + 100 + 500) / 3 = 700 / 3 = 233.3333
    assert abs(data3["mae"] - (700.0 / 3)) < 1e-3
    # RMSE = sqrt((10000 + 10000 + 250000) / 3) = sqrt(90000) = 300.0
    assert data3["rmse"] == 300.0
    # Mean error = (100 - 100 - 500) / 3 = -500 / 3 = -166.6667
    assert abs(data3["mean_error"] - (-500.0 / 3)) < 1e-3
    # Inside count = 2 of 3, Coverage = 66.67%
    assert data3["inside_interval_count"] == 2
    assert data3["coverage_percentage"] == 66.67
