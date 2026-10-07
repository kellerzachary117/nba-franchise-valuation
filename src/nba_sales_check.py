# Usage: python src/nba_sales_check.py data/panel.csv
import pandas as pd, sys
df=pd.read_csv(sys.argv[1]); fv=df.set_index(['team','season']).forbes_value
# (team, sale price $M, agreed date, Forbes list in force at agreement -> season label, control?)
sales=[('Minnesota Timberwolves',1500,'May 2021',2021,'Control'),   # Feb 2021 list
       ('Phoenix Suns',4000,'Dec 2022',2023,'Control'),             # Oct 2022 list
       ('Milwaukee Bucks',3500,'Feb 2023',2023,'Minority'),
       ('Charlotte Hornets',3000,'Jun 2023',2023,'Control'),
       ('Dallas Mavericks',3500,'Dec 2023',2024,'Control'),         # Oct 2023 list
       ('Boston Celtics',6100,'Mar 2025',2025,'Control'),           # Oct 2024 list
       ('Los Angeles Lakers',10000,'Jun 2025',2025,'Control'),
       ('Portland Trail Blazers',4250,'Aug 2025',2025,'Control')]
t=pd.DataFrame(sales,columns=['team','price','agreed','list','type'])
t['forbes']=[fv[(r.team,r.list)] for r in t.itertuples()]
t['price_vs_forbes']=(t.price/t.forbes-1).round(3)
print(t.to_string(index=False))
c=t[t.type=='Control']
print("\ncontrol sales: median premium",round(c.price_vs_forbes.median(),3),"mean",round(c.price_vs_forbes.mean(),3),"n",len(c))
