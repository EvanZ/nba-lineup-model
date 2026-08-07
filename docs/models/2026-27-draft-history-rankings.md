---
last_updated: "2026-08-05"
---

# 2026-27 Drafted Rookie Rankings

These are preseason cold-start RAPM priors for the 60 players in the official 2026 NBA Draft History response. They are not retrospective player evaluations and they do not include undrafted free agents, two-way signings, or other late entrants.

The table is sortable by every column. The prior continuously blends a draft-profile RAPM rate with the chance that a first-year player finishes below 5% of team possession opportunities, using the pooled replacement token for that low-exposure state.

## Inputs

- Draft class: official NBA `drafthistory` response for 2026-27.
- Draft-rate and exposure-gate training end at 2025-26.
- Replacement token: -3.96 RAPM.
- Forward-state source: `artifacts/models/forward_exposure_gated_rapm/2025-26/forward-exposure-gated-rapm-2025-26-20260806T030613Z-cddcd0ae`.
- Active roster profiles come from direct NBA `commonteamroster` responses when the player is on a listed team roster. Any remaining missing bio field uses its frozen historical drafted-player reference profile.

See [Exposure-Gated Cold-Start Prior](exposure-gated-cold-start.md) for the underlying blend and [Fetch Draft History](../guides/fetch-draft-history.md) for the direct-source ingestion contract.

## Rankings

| Rank | Cold-start RAPM prior | Player | Roster team | Pos. | Age | Ht. | Wt. | Pick | Draft rate | P(low exposure) | Affiliation |
| ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | -0.46 | AJ Dybantsa | WAS | F | 19 | 6-9 | 210 | 1 | -0.38 | 2.3% | Brigham Young |
| 2 | -0.52 | Darryn Peterson | UTA | G | 19 | 6-6 | 205 | 2 | -0.44 | 2.2% | Kansas |
| 3 | -0.59 | Caleb Wilson | CHI | F | 20 | 6-10 | 215 | 4 | -0.46 | 3.9% | North Carolina |
| 4 | -0.65 | Cameron Boozer | MEM | F | 19 | 6-10 | 250 | 3 | -0.55 | 2.9% | Duke |
| 5 | -0.71 | Mikel Brown Jr. | BKN | G | 20 | 6-5 | 190 | 6 | -0.58 | 3.7% | Louisville |
| 6 | -0.75 | Yaxel Lendeborg | GSW | F | 23 | 6-9 | 250 | 11 | -0.46 | 8.2% | Michigan |
| 7 | -0.79 | Keaton Wagler | LAC | G | 19 | 6-6 | 175 | 5 | -0.68 | 3.4% | Illinois |
| 8 | -0.90 | Darius Acuff Jr. | SAC | G | 19 | 6-2 | 190 | 7 | -0.80 | 3.2% | Arkansas |
| 9 | -1.01 | Morez Johnson | DAL | F | 20 | 6-9 | 250 | 9 | -0.82 | 5.9% | Michigan |
| 10 | -1.02 | Kingston Flemings | ATL | G | 19 | 6-4 | 190 | 8 | -0.89 | 4.1% | Houston |
| 11 | -1.02 | Brayden Burries | MIL | G | 20 | 6-4 | 205 | 10 | -0.86 | 5.3% | Arizona |
| 12 | -1.22 | Bennett Stirtz | OKC | G | 22 | 6-4 | 180 | 16 | -0.91 | 10.4% | Iowa |
| 13 | -1.24 | Aday Mara | OKC | C | 21 | 7-2 | 255 | 12 | -0.89 | 11.2% | Michigan |
| 14 | -1.33 | Dailyn Swain | CHI | F | 21 | 6-7 | 220 | 15 | -1.03 | 10.3% | Texas |
| 15 | -1.48 | Hannes Steinbach | CHA | F | 20 | 6-11 |  | 14 | -1.16 | 11.3% | Washington |
| 16 | -1.53 | Nate Ament | MIL | F | 19 | 6-10 | 207 | 13 | -1.26 | 10.0% | Tennessee |
| 17 | -1.57 | Christian Anderson | CHA | G | 20 | 6-0 | 165 | 18 | -1.31 | 9.9% | Texas Tech |
| 18 | -1.68 | Ebuka Okorie | DET | G | 19 | 6-2 | 185 | 17 | -1.43 | 9.7% | Stanford |
| 19 | -1.83 | Allen Graves | TOR | F | 20 | 6-9 | 225 | 19 | -1.42 | 16.1% | Santa Clara |
| 20 | -1.84 | Zuby Ejiofor | ATL | F | 22 | 6-9 | 240 | 23 | -1.29 | 20.4% | St. John's (NY) |
| 21 | -1.94 | Labaron Philon | PHI | G | 20 | 6-4 | 177 | 22 | -1.53 | 16.9% | Alabama |
| 22 | -1.95 | Tarris Reed Jr. | SAS | C | 23 | 6-11 | 265 | 26 | -1.27 | 25.1% | Connecticut |
| 23 | -1.97 | Cameron Carr | LAL | G | 21 | 6-5 | 175 | 24 | -1.47 | 20.2% | Baylor |
| 24 | -1.97 | Richie Saunders | MEM | G | 24 | 6-5 | 200 | 32 | -1.28 | 25.9% | Brigham Young |
| 25 | -2.04 | Jayden Quaintance | SAS | F | 19 | 6-10 | 255 | 20 | -1.64 | 17.6% | Kentucky |
| 26 | -2.06 | Alex Karaban | SAC | F | 23 | 6-8 | 220 | 29 | -1.36 | 26.9% | Houston |
| 27 | -2.09 | Bruce Thornton | HOU | G | 22 | 6-2 | 215 | 31 | -1.53 | 23.2% | Ohio State |
| 28 | -2.09 | Karim Lopez | MEM | F | 19 | 6-8 | 210 | 21 | -1.67 | 18.5% | Breakers (New Zealand) |
| 29 | -2.16 | Joshua Jefferson | BKN | F | 22 | 6-9 | 240 | 28 | -1.48 | 27.2% | Iowa State |
| 30 | -2.17 | Sergio de Larrea | DAL | G | 20 | 6-5 | 175 | 25 | -1.66 | 22.3% | Valencia BC (Spain) |
| 31 | -2.22 | Braden Smith | IND | G | 23 | 6-0 | 170 | 38 | -1.52 | 28.8% | Purdue |
| 32 | -2.41 | Otega Oweh | OKC | G | 23 | 6-4 | 215 | 41 | -1.58 | 35.2% | Kentucky |
| 33 | -2.43 | Ryan Conwell | MIA | G | 22 | 6-4 | 215 | 37 | -1.67 | 33.4% | Louisville |
| 34 | -2.44 | Trevon Brazile | DEN | F | 23 | 6-10 | 220 | 35 | -1.53 | 37.4% | Arkansas |
| 35 | -2.47 | Jaron Pierre Jr. | NOP | G | 24 | 6-5 | 210 | 58 | -1.31 | 43.9% | Southern Methodist |
| 36 | -2.52 | Ja'Kobi Gillespie | SAS | G | 22 | 6-1 | 186 | 42 | -1.71 | 36.3% | Tennessee |
| 37 | -2.53 | Bryce Hopkins | DEN | Unknown | 24 |  |  | 49 | -1.46 | 42.9% | St. John's (NY) |
| 38 | -2.57 | Chris Cenac Jr. | BOS | F | 19 | 6-11 | 240 | 27 | -1.93 | 31.6% | Houston |
| 39 | -2.65 | Koa Peat | PHX | F | 19 | 6-8 | 235 | 30 | -2.00 | 33.2% | Arizona |
| 40 | -2.65 | Emanuel Sharp | SAC | G | 22 | 6-3 | 210 | 45 | -1.73 | 41.4% | Houston |
| 41 | -2.67 | Isaiah Evans | MIN | F | 20 | 6-6 | 175 | 33 | -1.91 | 36.9% | Duke |
| 42 | -2.67 | Meleek Thomas | CLE | G | 20 | 6-5 | 185 | 34 | -1.92 | 36.6% | Arkansas |
| 43 | -2.68 | Baba Miller | LAC | F | 22 | 6-11 | 204 | 36 | -1.71 | 43.5% | Cincinnati |
| 44 | -2.72 | Dillon Mitchell | BOS | F | 22 | 6-8 | 205 | 40 | -1.74 | 44.1% | St. John's (NY) |
| 45 | -2.74 | Tobi Lawal | DAL | F | 23 | 6-8 | 215 | 48 | -1.62 | 48.2% | Virginia Tech |
| 46 | -2.76 | Jaden Bradley | TOR | G | 22 | 6-3 | 200 | 50 | -1.71 | 46.8% | Arizona |
| 47 | -2.82 | Tyler Bilodeau | BKN | F | 22 | 6-9 | 230 | 43 | -1.77 | 48.0% | California-Los Angeles |
| 48 | -2.82 | Jack Kayil | NYK | G | 20 | 6-3 | 174 | 39 | -1.99 | 42.3% | Alba Berlin (Germany) |
| 49 | -2.84 | Trey Kaufman-Renn | MIN | F | 23 | 6-9 | 230 | 59 | -1.45 | 55.6% | Purdue |
| 50 | -2.85 | Tyler Nickel | NYK | F | 22 | 6-7 | 220 | 47 | -1.75 | 49.6% | Vanderbilt |
| 51 | -2.85 | Maliq Brown | SAS | F | 22 | 6-9 | 225 | 44 | -1.77 | 49.5% | Duke |
| 52 | -2.94 | Lajae Jones | GSW | G-F | 22 | 6-7 | 220 | 54 | -1.68 | 55.6% | Florida State |
| 53 | -2.95 | Nick Martinelli | LAC | F | 22 | 6-7 | 220 | 55 | -1.66 | 56.2% | Northwestern |
| 54 | -2.97 | Felix Okpara | WAS | C | 22 | 6-11 | 235 | 46 | -1.78 | 54.4% | Tennessee |
| 55 | -3.01 | Vsevolod Ishchenko | DAL | G | 21 | 6-3 |  | 56 | -1.75 | 57.0% | Lokomotiv Kuban (Russia) |
| 56 | -3.03 | Izaiyah Nelson | ORL | F | 22 | 6-10 | 218 | 51 | -1.74 | 58.3% | South Florida |
| 57 | -3.11 | Henri Veesaar | ATL | C | 22 | 7-0 | 225 | 52 | -1.74 | 61.9% | North Carolina |
| 58 | -3.13 | Narcisse Ngoy | LAC | C | 22 | 7-0 |  | 57 | -1.65 | 64.2% | Poitiers Basket 86 (France) |
| 59 | -3.21 | Malique Lewis | MIL | F | 21 | 6-8 | 195 | 60 | -1.68 | 67.4% | South East Melbourne Magic (Australia) |
| 60 | -3.24 | Ugonna Onyenso | DET | C | 21 | 6-11 | 225 | 53 | -1.86 | 65.7% | Virginia |
