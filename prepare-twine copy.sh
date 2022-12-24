rm -r dist 
bash local-ci.sh
git push && git push codeberg
python3 -m build
twine check dist/*
#python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*
git push codeberg