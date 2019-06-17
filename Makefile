.PHONY : install

install:
	conda install --file requirements.txt --yes
	pip install -r requirements_pip.txt
	python setup.py develop
	pip install -r requirements_test.txt

lint:
	pytest --flake8
