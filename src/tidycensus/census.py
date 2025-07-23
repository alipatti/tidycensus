from functools import reduce
import json
from pathlib import Path
from typing import Any, Optional, get_args
from collections.abc import Mapping, Sequence
import os

import requests
from rich import print
import polars as pl
from polars import selectors as cs
from requests_cache import CachedSession


from tidycensus.types import Geography, AcsVersion, Dataset

BASE_API_URL = "https://api.census.gov/data/{year}/{dataset}"

MOST_RECENT_ACS_YEAR = 2023

# TODO: add docstrings

# TODO: add examples to readme


def _geo_dependenceis(geo: Geography) -> tuple[Geography, ...]:
    if geo == "county":
        return ("state", "county")

    return (geo,)


def _df_from_api_response(response: list[list[Any]]) -> pl.DataFrame:
    return pl.from_records(response[1:], schema=response[0], orient="row")


def _arrange_columns(df: pl.DataFrame) -> pl.DataFrame:
    all_columns = [
        "year",
        *get_args(Geography),
        "concept",
        "label",
        "variable",
        "value",
        "se",
    ]

    return df.select(c for c in all_columns if c in df.columns)


class Census:
    _api_key: Optional[str]
    session: CachedSession | requests.Session

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[Path] = Path("~/.cache/tidycensus/cache.sqlite").expanduser(),
        **kwargs,
    ):
        # take api key from parameter, then environment, then omit
        self._api_key = api_key or os.environ.get("CENSUS_API_KEY") or None

        if not self._api_key:
            print("[orange]Unable to find Census API key in the environment.")

        self.session = (
            CachedSession(cache, **kwargs) if cache else requests.Session(**kwargs)
        )

    def _api_req(self, url: str, params: dict[str, Any] = {}):
        # add api key to parameters
        if self._api_key:
            params = params | {"key": self._api_key}

        response = self.session.get(url, params=params)

        if not response.ok:
            print("[red][bold] --- REQUEST FAILED --- ")
            print(f"[red]{response.url}")

            raise RuntimeError("Unexpected response from Census API.")

        return json.loads(response.content)

    def get_metadata(
        self,
        dataset: Dataset,
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
        dataset: Dataset,
        *,
        years: Sequence[int],
        variables: Sequence[str] = [],
        geography: Geography = "us",
        filter: Mapping[Geography, str] = {},
        include_metadata=True,
    ) -> pl.DataFrame:
        " ".join(f"{k}:{v}" for k, v in filter.items())

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
                pl.col("value").cast(pl.Float32, strict=False),
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
        acs_version: AcsVersion = "acs5",
        geography: Geography = "us",
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
