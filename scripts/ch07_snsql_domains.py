import pandas as pd
import snsql

data, metadata = pd.read_csv("domain.csv"), "domain.yaml"
per_query_budget = snsql.Privacy(epsilon=0.5)

# reader is a SQL API over a privacy odometer
reader = snsql.from_df(data, privacy=per_query_budget, metadata="domain.yaml")

res = reader.execute_df("""
    SELECT domain, COUNT(domain) AS DomainVisits
    FROM MySchema.MyTable GROUP BY domain""")

print(res.sort_values(by="DomainVisits", ascending=False))


domains_visits = reader.execute_df('''
    SELECT Day_id, COUNT(*) AS DomainVisits
    FROM MySchema.MyTable GROUP BY Day_id''')

user_counts = reader.execute_df('''
    SELECT Day_id, COUNT(DISTINCT ids_num) AS Users
    FROM MySchema.MyTable GROUP BY Day_id''')