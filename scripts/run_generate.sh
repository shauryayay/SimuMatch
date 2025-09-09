'''#!/usr/bin/env bash
python src/synthetic/faker_profiles.py
python src/preprocessing/feature_engineering.py'''

#!/usr/bin/env bash
export PYTHONPATH=$(pwd)
python src/synthetic/faker_profiles.py
python src/preprocessing/feature_engineering.py