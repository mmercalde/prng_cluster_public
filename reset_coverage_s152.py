#!/usr/bin/env python3
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
deleted = conn.execute('''
    DELETE FROM exhaustive_progress
    WHERE prng_type = 'java_lcg'
    AND seed_range_start >= 660000000
''')
conn.commit()
print(f'Deleted {deleted.rowcount} rows')
row = conn.execute(
    'SELECT MAX(seed_range_end) FROM exhaustive_progress WHERE prng_type=?',
    ('java_lcg',)
).fetchone()
print(f'Coverage pointer now at: {row[0]:,}')
conn.close()
