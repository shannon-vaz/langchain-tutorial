import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")


def list_tables():
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        rows = cursor.fetchall()
        return "\n".join(row[0] for row in rows if row[0] is not None)
    except sqlite3.OperationalError as e:
        return f"The following error occurred: {str(e)}"
    finally:
        cursor.close()


def run_sqlite_query(query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as e:
        return f"The following error occurred: {str(e)}"
    finally:
        cursor.close()


def describe_tables(table_names: list[str]):
    cursor = conn.cursor()
    tables = ", ".join(f"'{t}'" for t in table_names)
    rows = cursor.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});"
    )
    return "\n".join(row[0] for row in rows if row[0] is not None)


run_query_tool = Tool.from_function(
    name="sqlite_query", description="Run a sqlite query", func=run_sqlite_query
)

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
)
