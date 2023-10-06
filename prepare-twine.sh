rm -r dist 
python3 -m build
python3 -m twine upload dist/*
git push
git push codeberg