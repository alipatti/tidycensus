from __future__ import annotations

from functools import cache
import json
from typing import Any, Callable, Literal, Optional, Sequence, TypeAlias
import requests
import os

from rich import print
import polars as pl
from joblib import Memory

# TODO: add more surveys
DATASET: TypeAlias = Literal["acs/acs5", "dec/sf3"] | str

# TODO: add more geographies
GEOGRAPHY: TypeAlias = Literal["us", "region", "division", "state", "county"]

# base url for census api
BASE_URL = "https://api.census.gov/data/{year}/{dataset}"


def _df_from_api_response(response: list[list[Any]]) -> pl.DataFrame:
    return pl.from_records(response[1:], schema=response[0], orient="row")


def _fetch(url: str, params: dict[str, Any]):

    response = requests.get(url, params=params)

    if not response.ok:
        print("[red][bold] --- REQUEST FAILED --- ")
        print(f"[red]{response.url}")
        raise RuntimeError("Unexpected response from Census API.")

    return response.content


class Census:

    _api_key: Optional[str]
    _fetch: Callable[[str, dict[str, Any]], Any]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_directory: Optional[str] = "/tmp/census-api-cache",
    ):
        # take api key from parameter, then environment, then omit
        self._api_key = api_key or os.environ.get("CENSUS_API_KEY") or None

        # cache the api responses if cache_directory is set, otherwise don't
        self._fetch = (
            Memory(cache_directory, verbose=1).cache(_fetch)
            if cache_directory
            else _fetch
        )

        if not self._api_key:
            print("[orange]Unable to find Census API key in the environment.")

    def _api_req(self, url: str, params: dict[str, Any] = {}):

        if self._api_key:
            params = params | {"key": self._api_key}

        return json.loads(self._fetch(url, params))

    @cache
    def _get_variable_info(
        self,
        dataset: DATASET,
        year: int,
    ):

        url = BASE_URL.format(year=year, dataset=dataset) + f"/variables.json"
        variables = self._api_req(url).get("variables")

        df = pl.from_records(
            [{"variable": k} | v for k, v in variables.items()],
            orient="row",
        )

        return (
            df.filter(pl.col("predicateOnly").is_null())
            .select(
                "variable",
                "concept",
                pl.col("label").str.split("!!"),
            )
            .sort(pl.col("variable"))
        )

    def _get_variables(
        self,
        dataset: DATASET,
        years: Sequence[int],
        variables: Sequence[str],
        geography: GEOGRAPHY = "us",
    ) -> pl.DataFrame:

        params = {
            "get": ",".join(variables),
            "for": f"{geography}:*",
        }

        # construct endpoint urls
        urls = [BASE_URL.format(year=year, dataset=dataset) for year in years]

        # fetch api responses
        responses = [self._api_req(url, params) for url in urls]

        # get variable labels/values

        # convert to dataframe
        estimates = (
            pl.concat(
                _df_from_api_response(response).with_columns(year=year)
                for year, response in zip(years, responses)
            )
            .unpivot(on=variables, index=["year", geography])
            .with_columns(
                pl.col("state").str.to_integer(),
                # TODO: deal with exception values
                pl.col("value").cast(pl.Float32),
            )
        )

        metadata = self._get_variable_info(dataset, years[0])

        return estimates.join(
            metadata,
            on="variable",
            how="left",
            validate="m:1",
        ).select(
            "year",
            geography,
            "concept",
            "label",
            "variable",
            "value",
        )


if __name__ == "__main__":
    # TODO: move to tests
    api = Census()

    variables = ["B19013_001E", "B19013A_001E"]
    geography = "state"
    dataset = "acs/acs5"
    years = [2010, 2012]

    df = api._get_variables(dataset, years, variables, geography=geography)


# response = requests.get(url, params).json()
# df = pl.from_records(response[1:], schema=response[0], orient="row")
