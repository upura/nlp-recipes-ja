## Dataset

This submodule includes helper functions for downloading datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

## Data Loading

There are dataloaders for several datasets. For example, the livedoor module will allow you to load a dataframe in pandas from the livedoor dataset, with the option to set the number of rows to load in order to test algorithms and evaluate performance benchmarks.
Most datasets may be split into `train`, `valid`, and `test`, for example:

```python
from utils_nlp.dataset.livedoor import load_pandas_df

df = load_pandas_df(nrows=1000, shuffle=False)
```

## Dataset List
|Dataset|Dataloader script|
|-------|-----------------|
|[livedoor ニュースコーパス](https://www.rondhuit.com/download.html)|[livedoor.py](./livedoor.py)|
