import sqlite3
conn = sqlite3.connect('c:/ai-trading-system/data/trading_state.db')
c = conn.cursor()
c.execute('SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 5')
rows = c.fetchall()
for row in rows:
    print(row)
conn.close()

