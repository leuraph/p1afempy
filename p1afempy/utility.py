import numpy as np


def is_row_in(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    identifies and returns the indices the rows in `a` that are also in `b`

    Parameters
    ----------
    a: np.ndarray
        the input array of which to identify rows shared with `b`
    b: np.ndarray
        the input array for which to check `a`'s rows against

    Returns
    -------
    np.ndarray:
        the indices of the rows of `a` shared with `b`

    References
    ----------
    implementation suggested in this answer https://stackoverflow.com/a/71394258
    """
    # identify the unique rows in `a` and `b` together 
    _, rev = np.unique(np.concatenate((b, a)), axis=0, return_inverse=True)

    # separate the inverse array into the part
    # corresponding to `a` and `b`, respectively
    a_rev = rev[len(b):]
    b_rev = rev[:len(b)]

    # where they share the same inverse index,
    # they point to the same unique row
    return np.isin(a_rev, b_rev)
