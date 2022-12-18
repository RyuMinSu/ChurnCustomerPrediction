import pandas as pd
import pymysql
from sqlalchemy import create_engine

def createTb(dfpath, user, pw, host, dbName, tbName):
    df = pd.read_csv(dfpath)
    dbConnpath = f"mysql+pymysql://{user}:{pw}@{host}/{dbName}"
    dbConn = create_engine(dbConnpath)
    conn = dbConn.connect()
    df.to_sql(name=tbName, con=conn, if_exists="fail", index=False)
    print(f"success! create {tbName} table in {dbName}")

# #db > df
def readTb(host, user, pw, dbName, tbName):
    conn = pymysql.connect(host=host, user=user, passwd=str(pw), db=dbName, charset="utf8")
    cur = conn.cursor()
    rsql = f"select * from {tbName}"
    df = pd.read_sql(rsql, con=conn)
    print(f"success! read {tbName} table in {dbName}")
    print(f"df shape: {df.shape}")
    return df