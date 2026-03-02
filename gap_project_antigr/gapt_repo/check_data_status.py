import os

stations = ['ABZ', 'ALE', 'AMA', 'AMM', 'ASP', 'BEI', 'BOT', 'BSL', 'DEL', 'EGB',
            'FKL', 'HAD', 'HEL', 'HPB', 'HRW', 'HYY', 'KCE', 'KPZ', 'MAR', 'MHD',
            'MLP', 'MUK', 'NAN', 'NEU', 'POV', 'SAO', 'SCH', 'SGP', 'UAE', 'PRL',
            'VAR', 'VHL', 'VIE', 'WAL', 'ZOT']

data_dir = 'data/N100_proxy'
missing = []
present = []

print("Checking for data files in", data_dir)
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exist!")
    exit(1)

for s in stations:
    filename = f"{s}_N100.dat"
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        present.append(s)
    else:
        missing.append(s)

print(f"\nFound data for {len(present)}/{len(stations)} stations.")
if missing:
    print("\nMissing data for:")
    for m in missing:
        print(f"- {m}")
    print("\nPlease download the N100 data from EBAS (http://ebas.nilu.no) and save as [CODE]_N100.dat")
    print("Refer to STATION_MAPPING.md for details.")
else:
    print("\nAll data present!")
