from typing import Tuple

import pytest

from release import matches_start, Version


def test_matches_start():
    assert matches_start('m', 'major')
    assert matches_start('ma', 'major')
    assert matches_start('maj', 'major')
    assert matches_start('majo', 'major')
    assert matches_start('major', 'major')
    assert not matches_start('major1', 'major')
    assert not matches_start('mi', 'major')
    assert not matches_start('mijor', 'major')


@pytest.mark.parametrize('string', ['67.0.2', '0.0.0', '1.2.0'])
def test_string_round_trip(string: str):
    assert str(Version.from_string(string)) == string


@pytest.mark.parametrize('string', ['67.0.2', '0.0.0', '1.2.0'])
def test_tag(string: str):
    assert Version.from_string(string).tag == 'v' + string


@pytest.mark.parametrize('string, numbers', [('67.0.2', (67, 0, 2)),
                                             ('1.1.1', (1, 1, 1)),
                                             ('a.b.c', ()),
                                             ('-1.0.0', ())])
def test_from_string(string: str, numbers: Tuple):
    if not numbers:
        with pytest.raises(ValueError):
            _ = Version.from_string(string)
    else:
        version = Version.from_string(string)
        assert version.tuple == numbers
        assert Version(*numbers) == version


@pytest.mark.parametrize('inputs, outputs', [((1, 3, 5), ((2, 0, 0), (1, 4, 0), (1, 3, 6))),
                                             ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))])
def test_increments(inputs: Tuple, outputs: Tuple):
    version = Version(*inputs)
    version.increment_major()
    assert version.tuple == outputs[0]

    version = Version(*inputs)
    version.increment_minor()
    assert version.tuple == outputs[1]

    version = Version(*inputs)
    version.increment_patch()
    assert version.tuple == outputs[2]


@pytest.mark.parametrize('inputs', [(-1, 0, 0), (0, -10, 0), (0, 0, -2), (0, -1, -2), (-1, 0, -2)])
def test_creation_errors(inputs: Tuple):
    with pytest.raises(ValueError):
        _ = Version(*inputs)
