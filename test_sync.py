"""
Sync module tests — validates the Seismo API contract.

Tests that payloads sent by sync.py match what Seismo's PHP expects,
and that HTTP errors propagate instead of being silently swallowed.
Uses unittest.mock to intercept httpx calls (no real network).
"""
import os
import sys
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Setup: use a temp DB so we don't pollute the real one ──
_test_dir = tempfile.mkdtemp(prefix="magnitu_test_sync_")
os.environ["MAGNITU_TEST"] = "1"

import config
config.DB_PATH = Path(_test_dir) / "test.db"
config.MODELS_DIR = Path(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = Path(_test_dir) / "test_config.json"

test_config = dict(config.DEFAULTS)
test_config["seismo_url"] = "https://test.example.com/index.php"
test_config["api_key"] = "test_key_123"
config.save_config(test_config)

import db
db.DB_PATH = config.DB_PATH

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    print("  TEST: {}".format(name), end=" ... ")
    return name


def ok():
    global PASS
    PASS += 1
    print("OK")


def fail(msg):
    global FAIL
    FAIL += 1
    ERRORS.append(msg)
    print("FAIL: {}".format(msg))


def make_mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = json.dumps(json_data or {})
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP {}".format(status_code),
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ═══════════════════════════════════════════
#  1. Label payload shape matches Seismo contract
# ═══════════════════════════════════════════
print("\n=== 1. Label Push Payload ===")

t = test("push_labels sends correct JSON structure")
try:
    db.set_label("feed_item", 100, "important", reasoning="test reason")
    db.set_label("email", 200, "noise", reasoning="")

    captured_kwargs = {}

    def capture_request(method, url, params=None, **kwargs):
        captured_kwargs.update({"method": method, "params": params, "kwargs": kwargs})
        return make_mock_response(200, {"success": True, "inserted": 2, "updated": 0, "errors": 0, "total": 2})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = capture_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        import sync
        sync.push_labels()

    payload = captured_kwargs["kwargs"]["json"]
    assert "labels" in payload, "Payload must have 'labels' key"
    assert isinstance(payload["labels"], list), "labels must be a list"
    for lbl in payload["labels"]:
        assert "entry_type" in lbl, "Each label must have entry_type"
        assert "entry_id" in lbl, "Each label must have entry_id"
        assert "label" in lbl, "Each label must have label"
        assert "reasoning" in lbl, "Each label must have reasoning"
        assert "labeled_at" in lbl, "Each label must have labeled_at"
    ok()
except Exception as e:
    fail(str(e))


t = test("push_labels entry_type values match Seismo ENUM")
try:
    for lbl in captured_kwargs["kwargs"]["json"]["labels"]:
        assert lbl["entry_type"] in ("feed_item", "email", "lex_item"), \
            "entry_type '{}' not in Seismo ENUM".format(lbl["entry_type"])
    ok()
except Exception as e:
    fail(str(e))


t = test("push_labels label values are valid classes")
try:
    valid = {"investigation_lead", "important", "background", "noise"}
    for lbl in captured_kwargs["kwargs"]["json"]["labels"]:
        assert lbl["label"] in valid, \
            "label '{}' not in valid classes".format(lbl["label"])
    ok()
except Exception as e:
    fail(str(e))


t = test("push_labels sends api_key in params")
try:
    assert captured_kwargs["params"]["api_key"] == "test_key_123", \
        "API key not sent or wrong"
    assert captured_kwargs["params"]["action"] == "magnitu_labels", \
        "Action should be magnitu_labels"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  2. Score payload shape matches Seismo contract
# ═══════════════════════════════════════════
print("\n=== 2. Score Push Payload ===")

t = test("push_scores sends correct JSON structure")
try:
    scores = [
        {
            "entry_type": "feed_item",
            "entry_id": 100,
            "relevance_score": 0.87,
            "predicted_label": "investigation_lead",
            "explanation": {
                "top_features": [{"feature": "keyword", "weight": 0.4, "direction": "positive"}],
                "confidence": 0.87,
                "prediction": "investigation_lead",
            },
        }
    ]

    captured_score = {}

    def capture_score_request(method, url, params=None, **kwargs):
        captured_score.update({"params": params, "kwargs": kwargs})
        return make_mock_response(200, {"success": True})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = capture_score_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        sync.push_scores(scores, model_version=7)

    payload = captured_score["kwargs"]["json"]
    assert "scores" in payload, "Payload must have 'scores' key"
    assert "model_version" in payload, "Payload must have 'model_version'"
    assert payload["model_version"] == 7
    s = payload["scores"][0]
    for field in ("entry_type", "entry_id", "relevance_score", "predicted_label"):
        assert field in s, "Score missing required field: {}".format(field)
    ok()
except Exception as e:
    fail(str(e))


t = test("push_scores entry_type values match Seismo ENUM")
try:
    for s in captured_score["kwargs"]["json"]["scores"]:
        assert s["entry_type"] in ("feed_item", "email", "lex_item"), \
            "entry_type '{}' not in Seismo ENUM".format(s["entry_type"])
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  3. Recipe payload shape matches Seismo contract
# ═══════════════════════════════════════════
print("\n=== 3. Recipe Push Payload ===")

t = test("push_recipe sends correct JSON structure")
try:
    recipe = {
        "version": 7,
        "classes": ["investigation_lead", "important", "background", "noise"],
        "class_weights": [1.0, 0.66, 0.33, 0.0],
        "keywords": {"corruption": {"investigation_lead": 0.5}},
        "source_weights": {"rss": {"important": 0.1}},
    }

    captured_recipe = {}

    def capture_recipe_request(method, url, params=None, **kwargs):
        captured_recipe.update({"params": params, "kwargs": kwargs})
        return make_mock_response(200, {"success": True})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = capture_recipe_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        sync.push_recipe(recipe)

    payload = captured_recipe["kwargs"]["json"]
    for field in ("version", "classes", "class_weights", "keywords"):
        assert field in payload, "Recipe missing required field: {}".format(field)
    assert payload["classes"] == ["investigation_lead", "important", "background", "noise"], \
        "Recipe classes must be the 4 standard labels in order"
    assert len(payload["class_weights"]) == 4, "class_weights must have 4 entries"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  4. Error propagation — HTTP errors are NOT swallowed
# ═══════════════════════════════════════════
print("\n=== 4. Error Propagation ===")

t = test("push_labels raises on HTTP 500")
try:
    # Clear sync log so push_labels will attempt to push all labels
    conn = db.get_db()
    conn.execute("DELETE FROM sync_log")
    conn.commit()
    conn.close()

    def fail_request(method, url, params=None, **kwargs):
        return make_mock_response(500, {"error": "Internal Server Error"})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = fail_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        raised = False
        try:
            sync.push_labels()
        except Exception:
            raised = True
        assert raised, "push_labels must raise on HTTP 500, not swallow the error"
    ok()
except Exception as e:
    fail(str(e))


t = test("push_scores raises on HTTP 500")
try:
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = fail_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        raised = False
        try:
            sync.push_scores([{"entry_type": "feed_item", "entry_id": 1, "relevance_score": 0.5, "predicted_label": "noise"}], 1)
        except Exception:
            raised = True
        assert raised, "push_scores must raise on HTTP 500"
    ok()
except Exception as e:
    fail(str(e))


t = test("pull_labels raises on HTTP 500")
try:
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = fail_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        raised = False
        try:
            sync.pull_labels()
        except Exception:
            raised = True
        assert raised, "pull_labels must raise on HTTP 500"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  5. Pull labels merge logic
# ═══════════════════════════════════════════
print("\n=== 5. Pull Labels Merge ===")

t = test("pull_labels imports new labels")
try:
    # Clear existing labels
    conn = db.get_db()
    conn.execute("DELETE FROM labels")
    conn.commit()
    conn.close()

    remote_labels = {
        "labels": [
            {"entry_type": "feed_item", "entry_id": 500, "label": "important", "reasoning": "remote", "labeled_at": "2026-01-01 12:00:00"},
        ],
        "total": 1,
    }

    def pull_request(method, url, params=None, **kwargs):
        return make_mock_response(200, remote_labels)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = pull_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        count = sync.pull_labels()

    assert count == 1, "Expected 1 imported label, got {}".format(count)
    local = db.get_label_with_reasoning("feed_item", 500)
    assert local is not None, "Label should be in local DB"
    assert local["label"] == "important"
    assert local["reasoning"] == "remote"
    ok()
except Exception as e:
    fail(str(e))


t = test("pull_labels: remote newer wins conflict")
try:
    db.set_label("feed_item", 600, "noise", reasoning="old local")
    remote_labels = {
        "labels": [
            {"entry_type": "feed_item", "entry_id": 600, "label": "important", "reasoning": "newer remote", "labeled_at": "2099-01-01 00:00:00"},
        ],
        "total": 1,
    }

    def pull_request(method, url, params=None, **kwargs):
        return make_mock_response(200, remote_labels)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = pull_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        sync.pull_labels()

    local = db.get_label_with_reasoning("feed_item", 600)
    assert local["label"] == "important", "Remote newer label should win, got '{}'".format(local["label"])
    assert local["reasoning"] == "newer remote"
    ok()
except Exception as e:
    fail(str(e))


t = test("pull_labels: local newer keeps local")
try:
    db.set_label("feed_item", 700, "investigation_lead", reasoning="local wins")
    remote_labels = {
        "labels": [
            {"entry_type": "feed_item", "entry_id": 700, "label": "noise", "reasoning": "old remote", "labeled_at": "2000-01-01 00:00:00"},
        ],
        "total": 1,
    }

    def pull_request(method, url, params=None, **kwargs):
        return make_mock_response(200, remote_labels)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = pull_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        sync.pull_labels()

    local = db.get_label_with_reasoning("feed_item", 700)
    assert local["label"] == "investigation_lead", "Local newer label should be kept, got '{}'".format(local["label"])
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  6. Verify endpoint smoke test function
# ═══════════════════════════════════════════
print("\n=== 6. Endpoint Verification ===")

t = test("verify_seismo_endpoints detects working endpoint")
try:
    def ok_request(method, url, params=None, **kwargs):
        return make_mock_response(200, {"success": True, "inserted": 0, "updated": 0, "errors": 0, "total": 0})

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = ok_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        success, msg = sync.verify_seismo_endpoints()
    assert success, "Should return True for healthy endpoint, got: {}".format(msg)
    ok()
except Exception as e:
    fail(str(e))


t = test("verify_seismo_endpoints detects broken endpoint")
try:
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.request.side_effect = fail_request
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client_cls.return_value = mock_client

        success, msg = sync.verify_seismo_endpoints()
    assert not success, "Should return False for broken endpoint"
    assert "label" in msg.lower() or "500" in msg, "Message should mention the issue"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════
print("\n" + "=" * 50)
print("Results: {} passed, {} failed".format(PASS, FAIL))
if ERRORS:
    print("\nFailures:")
    for err in ERRORS:
        print("  - {}".format(err))
print("=" * 50)

# Cleanup
shutil.rmtree(_test_dir, ignore_errors=True)

sys.exit(0 if FAIL == 0 else 1)
