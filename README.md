# tidycensus

A Python wrapper for the U.S. Census Bureau API. Shamelessly inspired by the
[tidycensus](https://github.com/walkerke/tidycensus) package in R.

## Installation 

```bash
pip install tidycensus # or uv add tidycensus
```

## Usage

You first have to find the name of the variable you're interested in.

I recommend searching on the [Census Bureau's data explorer](https://data.census.gov/table).

To get median income by county, you would search "median income" and discover that the corresponding code is "XXXX".

```python
# TODO: write example of fetching data

# TODO: example of fetching entire table (with a group)

# TODO: acs example (with counties)

# TODO: population example (with block groups / tracts) 
```

## Future work

- automatically extract the suffixes from variables and group them appropriately (e.g. E, M, AN, etc.)

- CLI?
    - `tidycensus fetch <var> --output incomes.parquet`
    - `tidycensus search "median income"`
        ```bash
        curl "https://data.census.gov/api/search?q=median%20income%20white&services=search&size=20&y=2020" \
          | jq ".response.tables.tables[] | {table, desc: .instances[].description}"
        ```
    - `tidycensus cache (clear | list | path)`

