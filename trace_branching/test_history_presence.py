from presence_semantics import adjusted_values


def test_union_semantics() -> None:
    values = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    got = adjusted_values(values, {0, 1}, {1, 2}, 1.5)
    assert got == {0: 8.5, 1: 8.5, 2: 8.5, 3: 10.0}


def test_empty_prefix_is_native_semantics() -> None:
    values = {0: 3.0, 1: 3.0}
    assert adjusted_values(values, set(), {1}, 1.5) == {0: 3.0, 1: 1.5}


if __name__ == "__main__":
    test_union_semantics()
    test_empty_prefix_is_native_semantics()
    print("history-presence specification tests passed")
