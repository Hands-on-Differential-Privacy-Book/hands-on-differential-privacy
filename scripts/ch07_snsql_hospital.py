import pandas as pd
import snsql


# metadata.yml has been configured with the per-user visit threshold
data, metadata = pd.read_csv("visits.csv"), "metadata.yml"
per_query_budget = snsql.Privacy(epsilon=0.5)

# reader is a SQL API over a privacy odometer
reader = snsql.from_df(data, privacy=per_query_budget, metadata=metadata)

total_visits = reader.execute(
    """SELECT SUM(visits) as TotalVisits FROM HospitalRecords.Visits""")

average_visits = reader.execute(
    """SELECT AVG(visits) as AverageVisits FROM HospitalRecords.Visits""")


reader = snsql.from_df(data, privacy=per_query_budget, metadata='events.yml')

event_counts = reader.execute_df('''
    SELECT Events, COUNT(Events) AS e
    FROM MySchema.MyTable GROUP BY Events''')