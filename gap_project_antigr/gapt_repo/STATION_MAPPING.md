# GapT Station Mapping

The following table maps the station codes found in the GapT repository (`notebooks/preprocess_data.ipynb`) to likely EBAS/GAW stations. 
**Note**: This mapping is inferred and may require verification.

| Code | Likely Station Name | Country | GAW ID (likely) |
|---|---|---|---|
| ABZ | Abu Dhabi? (Unclear) | ? | ? |
| ALE | Alert | Canada | ALE |
| AMA | Amazon (Manaus) / ATTO? | Brazil | ? |
| AMM | Amman? | Jordan | ? |
| ASP | Aspvreten | Sweden | ASP |
| BEI | Shangdianzi (Beijing) | China | SDZ (or BEI in GapT) |
| BOT | Botswana? | Botswana | ? |
| BSL | Basel? | Switzerland | ? |
| DEL | Delhi? | India | ? |
| EGB | Egbert | Canada | EGB |
| FKL | Finokalia | Greece | FKL |
| HAD | Harwell? | UK | HAR? |
| HEL | Helsinki (SMEAR III?) | Finland | ? |
| HPB | Hohenpeißenberg | Germany | HPB |
| HRW | ? | ? | ? |
| HYY | Hyytiälä (SMEAR II) | Finland | HYY |
| KCE | ? | ? | ? |
| KPZ | K-Puszta | Hungary | KPU? |
| MAR | Marambio | Antarctica (Argentina) | MAR? |
| MHD | Mace Head | Ireland | MHD |
| MLP | Melpitz | Germany | MEL? |
| MUK | ? | ? | ? |
| NAN | ? | ? | ? |
| NEU | Neumayer | Antarctica (Germany) | NEU |
| POV | ? | ? | ? |
| SAO | Sao Paulo? | Brazil | ? |
| SCH | Schauinsland | Germany | SSL? |
| SGP | Southern Great Plains | USA | SGP |
| UAE | United Arab Emirates? | UAE | ? |
| PRL | ? | ? | ? |
| VAR | Värriö (SMEAR I) | Finland | ? |
| VHL | ? | ? | ? |
| VIE | Vienna? | Austria | ? |
| WAL | Waldhof | Germany | WAL? |
| ZOT | Zotino | Russia | ZOT |

## Instructions
1. Go to [EBAS Data Access](http://ebas.nilu.no/Default.aspx).
2. Search for the station name (e.g., "Alert").
3. Select columns: **Matrix=pm10** (or pm25/aerosol), **Component=particle_number_concentration**.
4. Download the `.nas` or `.dat` file.
5. Rename the file to `[CODE]_N100.dat` (e.g., `ALE_N100.dat`).
6. Place it in `gapt_repo/data/N100_proxy/`.
