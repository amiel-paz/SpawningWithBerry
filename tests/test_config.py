from pathlib import Path

import pytest

from aims_berry.config import AU_TIME_PER_FS, ConfigError, load_config


def _write(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "quoted geometry.xyz").write_text("1\ncomment\nH 0 0 0\n")
    path = tmp_path / "aims.in"
    path.write_text(
        "provider custom # comment\n"
        'geometry "quoted geometry.xyz"\n'
        "num_states 2\ninitial_state 1\ntime_step 0.5 fs\nsimulation_time 2 au\n"
        "electronic_method custom\n"
        "provider_option label first\nprovider_option scale 2.5\n"
        "observable bond impossible_but_parsed 0 1\n" + extra
    )
    return path


def test_parser_comments_quoting_units_and_repeated_records(tmp_path):
    config = load_config(_write(tmp_path))
    assert config.geometry == tmp_path / "quoted geometry.xyz"
    assert config.time_step == pytest.approx(0.5 * AU_TIME_PER_FS)
    assert config.provider_option_dict() == {"label": "first", "scale": 2.5}
    assert config.observables[0].atoms == (0, 1)


@pytest.mark.parametrize("extra, match", [
    ("num_states 3\n", "duplicate"),
    ("not_a_keyword 4\n", "unknown keyword"),
    ("state_weights 0.2 banana\n", "could not convert"),
])
def test_parser_reports_filename_and_line(tmp_path, extra, match):
    path = _write(tmp_path, extra)
    with pytest.raises(ConfigError, match=match) as caught:
        load_config(path)
    assert str(path.resolve()) in str(caught.value)
