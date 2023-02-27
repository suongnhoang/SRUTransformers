#bin/bash
find . | grep -E "(semantic-similarity.log|.ipynb_checkpoints|__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
rm -rf ./results

