"""Object Array reader.

A parser for tabular object array using pandas.

"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.base import Document


class ObjectArrayReader(BaseReader):
    r"""An Object Array parser.

    A parser for tabular object arrays using pandas.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        col_joiner (str): Separator to use for joining cols per row.
            Set to ", " by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.DataFrame` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        col_joiner: str = ", ",
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self, file: Path | list[dict], extra_info: Optional[Dict] = {}
    ) -> List[Document]:
        """Parse file."""
        import pandas as pd
        import json
        if type(file) == str:
            with open(file, 'r') as f:
                object_array_str: str = json.load(f.read())
            df = pd.DataFrame(object_array_str, **self._pandas_config)
        if type(file) == list:
            df = pd.DataFrame(file, **self._pandas_config)

        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()
        df_metadata:dict = {
            "columns": str(df.columns.tolist()),
            "schema": str(df.info(verbose=True)),
            "shape": str(df.shape),
        }
        for k, v in df_metadata.items():
            extra_info[k] = v
        if self._concat_rows:
            return [
                Document(
                    text=self._row_joiner.join(text_list), extra_info=extra_info or {},
                    metadata=df_metadata,
                )
            ]
        else:
            return [
                Document(text=text, extra_info=extra_info or {}, metadata=df_metadata,) 
                for text in text_list
            ]