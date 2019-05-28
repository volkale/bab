.PHONY : install

install:
	conda install --file requirements.txt --yes
	pip install -r requirements_pip.txt
	python setup.py develop