from __future__ import annotations, with_statement

from functools import reduce
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal, Optional, Sequence, TypeAlias, get_args
import os

import requests
from rich import print
import polars as pl
from polars import selectors as cs
from joblib import Memory

# TODO: add more surveys
DATASET: TypeAlias = Literal["acs/acs5", "dec/sf3"] | str

# TODO: add more geographies
GEOGRAPHY: TypeAlias = Literal["us", "region", "division", "state", "county"]

ACS_VERSION: TypeAlias = Literal["acs1", "acs3", "acs5"]

BASE_API_URL = "https://api.census.gov/data/{year}/{dataset}"

MOST_RECENT_ACS_YEAR = 2023

# TODO: add support for getting entire groups of variables
#   (use the groups(...) parameter in the census api)

# TODO: add docstrings

# TODO: add examples directory


def _geo_dependenceis(geo: GEOGRAPHY) -> tuple[GEOGRAPHY, ...]:
    if geo == "county":
        return ("state", "county")

    return (geo,)


def _df_from_api_response(response: list[list[Any]]) -> pl.DataFrame:
    return pl.from_records(response[1:], schema=response[0], orient="row")


def _fetch(url: str, params: dict[str, Any]):
    response = requests.get(url, params=params)

    if not response.ok:
        print("[red][bold] --- REQUEST FAILED --- ")
        print(f"[red]{response.url}")
        raise RuntimeError("Unexpected response from Census API.")

    return response.content


def _arrange_columns(df: pl.DataFrame) -> pl.DataFrame:
    all_columns = [
        "year",
        *get_args(GEOGRAPHY),
        "concept",
        "label",
        "variable",
        "value",
        "se",
    ]

    return df.select(c for c in all_columns if c in df.columns)


class Census:
    _api_key: Optional[str]
    _fetch: Callable[[str, dict[str, Any]], Any]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_directory: Optional[Path] = Path("~/.cache/tidycensus").expanduser(),
        cache_verbosity: int = 0,
    ):
        # take api key from parameter, then environment, then omit
        self._api_key = api_key or os.environ.get("CENSUS_API_KEY") or None

        # cache the api responses if cache_directory is set, otherwise don't
        self._fetch = (
            Memory(cache_directory, verbose=cache_verbosity).cache(_fetch)
            if cache_directory
            else _fetch
        )

        if not self._api_key:
            print("[orange]Unable to find Census API key in the environment.")

    def _api_req(self, url: str, params: dict[str, Any] = {}):
        if self._api_key:
            params = params | {"key": self._api_key}

        return json.loads(self._fetch(url, params))

    def get_metadata(
        self,
        dataset: DATASET,
        years: int | Sequence[int],
    ):
        if not isinstance(years, int):
            return pl.concat(self.get_metadata(dataset, year) for year in years)

        url = BASE_API_URL.format(year=years, dataset=dataset) + "/variables.json"
        response = self._api_req(url).get("variables")

        return (
            pl.from_records(
                [{"variable": k} | v for k, v in response.items()],
                orient="row",
            )
            .filter(pl.col("predicateOnly").is_null())
            .select(
                pl.lit(years).alias("year"),
                "variable",
                "concept",
                pl.col("label").str.split("!!"),
            )
            .sort(pl.col("variable"))
        )

    def get_variables(
        self,
        dataset: DATASET,
        years: Sequence[int],
        variables: Sequence[str] = [],
        geography: GEOGRAPHY = "us",
        include_metadata=True,
    ) -> pl.DataFrame:
        params = {
            "get": ",".join(variables),
            "for": f"{geography}:*",
        }

        # get base name for all the groups
        is_variable = reduce(
            lambda x, y: x | y,
            [
                cs.starts_with(v.removeprefix("group(").removesuffix(")"))
                if v.startswith("group(")
                else cs.matches(v)
                for v in variables
            ],
        )

        # construct endpoint urls
        urls = [BASE_API_URL.format(year=year, dataset=dataset) for year in years]

        # fetch api responses
        responses = [self._api_req(url, params) for url in urls]

        # convert to dataframe
        estimates = (
            pl.concat(
                _df_from_api_response(response).with_columns(year=year)
                for year, response in zip(years, responses)
            )
            .with_columns(
                reduce(
                    lambda x, y: x + y,
                    (pl.col(g) for g in _geo_dependenceis(geography)),
                ).alias(geography)
            )
            .unpivot(on=is_variable, index=["year", geography])
            .with_columns(
                pl.col(geography).cast(pl.Categorical(ordering="lexical")),
                # TODO: deal with exception values
                pl.col("value").cast(pl.Float32),
            )
            .sort("year", geography, "variable")
        )

        if not include_metadata:
            return estimates.pipe(_arrange_columns)

        metadata = self.get_metadata(dataset, years)

        return estimates.join(
            metadata,
            on=["year", "variable"],
            how="left",
            validate="m:1",
        ).pipe(_arrange_columns)

    def acs(
        self,
        variables: Sequence[str],
        acs_version: ACS_VERSION = "acs5",
        geography: GEOGRAPHY = "us",
        years: Optional[Sequence[int]] = None,
        include_ses=True,
        include_metadata=True,
    ) -> pl.DataFrame:
        dataset = f"acs/{acs_version}"

        # if years isn't passed, use all available years
        years = years or range(2004 + int(acs_version[-1]), MOST_RECENT_ACS_YEAR + 1)

        # standardize acs variable names (make sure they all end in E)
        base_variable_names = {v.strip("EM") for v in variables}
        variables = [f"{v}E" for v in base_variable_names]

        # include margins of error
        if include_ses:
            variables += [f"{v}M" for v in base_variable_names]

        # call to API
        response = self.get_variables(
            dataset=dataset,
            variables=variables,
            years=years,
            geography=geography,
            include_metadata=False,
        )

        # reshape so moes and estimates are the same row
        df = (
            response.with_columns(
                pl.col("variable").str.strip_chars_end("EM"),
                pl.col("variable").str.extract("(E|M)$").alias("type"),
            )
            .pivot(
                on="type",
                values="value",
                index=["year", geography, "variable"],
                maintain_order=True,
            )
            .rename({"E": "value"})
        )

        if include_ses:
            df = df.with_columns(pl.col("M").mul(1 / 1.645).alias("se")).drop("M")

        if not include_metadata:
            return df.pipe(_arrange_columns)

        return df.join(
            self.get_metadata(dataset, years),
            how="left",
            left_on=["year", pl.col("variable") + "E"],
            right_on=["year", "variable"],
            validate="m:1",
        ).pipe(_arrange_columns)
