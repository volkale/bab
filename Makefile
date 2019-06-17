.PHONY : install

install:
	pip install -r requirements.txt
	python setup.py install
	pip install -r requirements_test.txt

lint:
	pytest bab --flake8
