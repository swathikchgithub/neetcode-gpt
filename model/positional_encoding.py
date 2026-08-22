import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pe = np.zeros((seq_len, d_model), dtype=np.float64)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                denominator = np.power(10000, ( 2 * ( i //2)) / d_model)

                pe[pos, i] = np.sin(pos / denominator)

                if i+ 1 < d_model:
                    pe[pos, i+1] = np.cos(pos / denominator)
        return np.round(pe, 5)
