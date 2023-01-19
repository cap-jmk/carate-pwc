# Copyright Julian M. Kleber 

echo "local-ci  Copyright (C) 2022 Julian M. Kleber This program comes with ABSOLUTELY NO WARRANTY; for details type 'show w'. This is free software, and you are welcome to redistribute it under certain conditions; type 'show c' for details."

black carate/
find . -type f -name 'carate/*.py' -exec sed --in-place 's/[[:space:]]\+$//' {} \+ #sanitize trailing whitespace
autopep8 --in-place --recursive carate/ -j -1
python -m flake8 carate/ --count --select=E9,F63,F7,F82 --show-source --statistics
mypy --strict carate/
python -m pylint -f parseable carate/
pytest tests/