import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


pd.set_option('display.max_columns', 20)
file_path = "airlines_flights_data.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "rohitgrewal/airlines-flights-data",
  file_path,
)

max_price = max(df['price'])

most_expensive_tickets = df[df['price'] == max_price]
print(most_expensive_tickets)