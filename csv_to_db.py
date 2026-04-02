import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:1234@localhost/olist')

base = '/Users/a2485/Desktop/adv/data/'

tables = {
    'customers':   'olist_customers_dataset.csv',
    'sellers':     'olist_sellers_dataset.csv',
    'products':    'olist_products_dataset.csv',
    'orders':      'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'order_payments': 'olist_order_payments_dataset.csv',
    'order_reviews':  'olist_order_reviews_dataset.csv',
    'product_category_name_translation': 'product_category_name_translation.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
}

for table, file in tables.items():
    df = pd.read_csv(base + file)
    df.to_sql(table, engine, if_exists='replace', index=False)
    print(f'{table}: {len(df)} rows 완료')