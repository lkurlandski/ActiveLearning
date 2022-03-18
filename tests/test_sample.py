from active_learning import processor


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5
