'''#!/usr/bin/env bash
uvicorn src.serving.api:app --reload --port 8000'''

#!/usr/bin/env bash
export PYTHONPATH=$(pwd)
uvicorn src.serving.api:app --reload --port 8000