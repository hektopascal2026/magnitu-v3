"""
Phase 1 — Gemini JSON repair, prompt validation, synthetic scorer (mocked HTTP).

No real network or API keys required.
"""
import sys
from unittest.mock import create_autospec

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


print("\n=== parse_json_lenient ===")

t = test("parses clean JSON object")
try:
    from magnitu.gemini import parse_json_lenient

    o = parse_json_lenient('{"label": "noise", "reasoning": "x"}')
    assert o["label"] == "noise"
    ok()
except Exception as e:
    fail(str(e))

t = test("unwraps markdown fence")
try:
    from magnitu.gemini import parse_json_lenient

    text = "```json\n{\"a\": 1}\n```"
    o = parse_json_lenient(text)
    assert o["a"] == 1
    ok()
except Exception as e:
    fail(str(e))

t = test("fixes trailing comma")
try:
    from magnitu.gemini import parse_json_lenient

    o = parse_json_lenient('{"label": "noise", "reasoning": "ok",}')
    assert o["label"] == "noise"
    ok()
except Exception as e:
    fail(str(e))


print("\n=== validate_synthetic_label_output ===")

t = test("accepts valid label and reasoning")
try:
    from magnitu.prompts import validate_synthetic_label_output

    label, reason = validate_synthetic_label_output(
        {"label": "important", "reasoning": "  Has substance.  "}
    )
    assert label == "important"
    assert reason == "Has substance."
    ok()
except Exception as e:
    fail(str(e))

t = test("rejects bad label")
try:
    from magnitu.prompts import validate_synthetic_label_output

    validate_synthetic_label_output({"label": "spam", "reasoning": "x"})
    fail("expected ValueError")
except ValueError:
    ok()
except Exception as e:
    fail(str(e))

t = test("rejects empty reasoning")
try:
    from magnitu.prompts import validate_synthetic_label_output

    validate_synthetic_label_output({"label": "noise", "reasoning": "  "})
    fail("expected ValueError")
except ValueError:
    ok()
except Exception as e:
    fail(str(e))


print("\n=== should_retry_investigation_lead_empty_reasoning ===")

t = test("true only for investigation_lead with blank reasoning")
try:
    from magnitu.prompts import should_retry_investigation_lead_empty_reasoning

    assert should_retry_investigation_lead_empty_reasoning(
        {"label": "investigation_lead", "reasoning": ""}
    )
    assert not should_retry_investigation_lead_empty_reasoning(
        {"label": "investigation_lead", "reasoning": "EU-only scope."}
    )
    assert not should_retry_investigation_lead_empty_reasoning(
        {"label": "noise", "reasoning": ""}
    )
    ok()
except Exception as e:
    fail(str(e))


print("\n=== call_gemini_for_synthetic_label (autospec client) ===")

t = test("single call when first response is valid")
try:
    from magnitu.gemini import GeminiClient
    from magnitu.gemini_config import GeminiConfig
    from magnitu.synthetic_scorer import call_gemini_for_synthetic_label

    # Autospec enforces request_json(..., system_instruction=...) matches GeminiClient.
    client = create_autospec(GeminiClient, instance=True)
    client.cfg = GeminiConfig()
    client.request_json.return_value = {
        "label": "background",
        "reasoning": "Routine coverage.",
    }
    label, reason = call_gemini_for_synthetic_label(client, title="T", content="Body")
    assert label == "background"
    assert reason == "Routine coverage."
    assert client.request_json.call_count == 1
    ok()
except Exception as e:
    fail(str(e))

t = test("second call when investigation_lead has empty reasoning first")
try:
    from magnitu.gemini import GeminiClient
    from magnitu.gemini_config import GeminiConfig
    from magnitu.synthetic_scorer import call_gemini_for_synthetic_label

    client = create_autospec(GeminiClient, instance=True)
    client.cfg = GeminiConfig()
    client.request_json.side_effect = [
        {"label": "investigation_lead", "reasoning": ""},
        {
            "label": "investigation_lead",
            "reasoning": "Mentions Drittstaaten treatment.",
        },
    ]
    label, reason = call_gemini_for_synthetic_label(client, title="T")
    assert label == "investigation_lead"
    assert "Drittstaaten" in reason
    assert client.request_json.call_count == 2
    ok()
except Exception as e:
    fail(str(e))

t = test("raises after retry still invalid")
try:
    from magnitu.gemini import GeminiClient
    from magnitu.gemini_config import GeminiConfig
    from magnitu.synthetic_scorer import call_gemini_for_synthetic_label

    client = create_autospec(GeminiClient, instance=True)
    client.cfg = GeminiConfig()
    client.request_json.side_effect = [
        {"label": "investigation_lead", "reasoning": ""},
        {"label": "investigation_lead", "reasoning": "   "},
    ]
    call_gemini_for_synthetic_label(client, title="T")
    fail("expected ValueError")
except ValueError:
    assert client.request_json.call_count == 2
    ok()
except Exception as e:
    fail(str(e))


print("\n=== library_catalog ===")

t = test("summarize_magnitu_package reads manifest and labels")
try:
    import json
    import tempfile
    import zipfile
    from pathlib import Path

    from magnitu.library_catalog import summarize_magnitu_package, safe_package_path

    td = Path(tempfile.mkdtemp())
    zpath = td / "demo.magnitu"
    manifest = {
        "manifest_format_version": 1,
        "model_name": "Demo",
        "model_uuid": "abc",
        "version": 3,
        "label_count": 2,
        "metrics": {"accuracy": 0.9, "f1": 0.85},
        "label_distribution": {"investigation_lead": 1, "important": 1, "background": 0, "noise": 0},
    }
    labels = [
        {"label": "noise", "reasoning": "x"},
        {"label": "important", "reasoning": "y"},
    ]
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("labels.json", json.dumps(labels))
        zf.writestr("model.joblib", b"fake")
    s = summarize_magnitu_package(zpath)
    assert s["model_name"] == "Demo"
    assert s["label_distribution"]["noise"] >= 1
    assert s["has_model_joblib"]
    resolved = safe_package_path(td, "demo.magnitu")
    assert resolved == zpath.resolve()
    ok()
except Exception as e:
    fail(str(e))


print("\n" + "=" * 50)
print("Results: {} passed, {} failed".format(PASS, FAIL))
if ERRORS:
    print("\nFailures:")
    for err in ERRORS:
        print("  - {}".format(err))
print("=" * 50)

sys.exit(0 if FAIL == 0 else 1)
