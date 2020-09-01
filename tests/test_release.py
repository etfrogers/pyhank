from release import matches_start


def test_matches_start():
    assert matches_start('m', 'major')
    assert matches_start('ma', 'major')
    assert matches_start('maj', 'major')
    assert matches_start('majo', 'major')
    assert matches_start('major', 'major')
    assert not matches_start('major1', 'major')
    assert not matches_start('mi', 'major')
    assert not matches_start('mijor', 'major')
