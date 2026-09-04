"""Dependency-free specification of the intended presence-penalty behavior."""


def adjusted_values(
    values: dict[int, float],
    prefix_ids: set[int],
    new_output_ids: set[int],
    penalty: float,
) -> dict[int, float]:
    return {
        token_id: value - (penalty if token_id in prefix_ids | new_output_ids else 0)
        for token_id, value in values.items()
    }
