import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import preprocess


def main():
    X, y = preprocess.fetch_raw_data()
    preprocess.preprocess_data(X, y)
    print("Data fetched and preprocessed successfully.")


if __name__ == "__main__":
    main()
