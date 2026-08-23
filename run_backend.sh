#!/bin/bash
cd "$(dirname "$0")/src/backend"
# Watch only the python that the API actually imports. Reloading on frontend edits would
# cost a full model reload (~8s) for a change the backend never sees.
uvicorn api:app --reload --reload-dir . --reload-dir ../model --host 0.0.0.0 --port 8000
