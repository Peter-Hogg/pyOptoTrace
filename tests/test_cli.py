import json

from lumembrane.cli import main


def test_demo_is_deterministic(capsys):
    return_code = main(["demo", "--duration", "2", "--seed", "4"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["version"] == "0.1.0a0"
    assert payload["injected_spikes"] == payload["detected_spikes"]
