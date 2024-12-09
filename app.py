import dash
#import dash_snapshots
import os

os.environ["REDIS_URL"] = os.getenv("REDIS_URL", os.getenv("EXTERNAL_REDIS_URL"))
os.environ["DATABASE_URL"] = os.getenv(
    "DATABASE_URL", os.getenv("EXTERNAL_DATABASE_URL")
)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Use DATABASE_URL as provided by Dash Enterprise
# If on Windows use the default Postgres URL
# Fallback to SQLite on Mac and Linux
# On Windows you’ll need to download and run Postgres as built in SQLite is not supported
os.environ["SNAPSHOT_DATABASE_URL"] = (
    os.environ.get("DATABASE_URL", "postgres://username:password@127.0.0.1:5432")
    if os.name == "nt"
    else os.environ.get("DATABASE_URL", "sqlite:///snapshot-dev.db")
)

#snap = dash_snapshots.DashSnapshots(app)
