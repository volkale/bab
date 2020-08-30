.PHONY: clean install

clean:
	rm -rf *.o
	rm -rf .cache
	rm -rf .eggs
	rm -rf *.egg-info
	rm -rf *.png
	rm -rf *.pkl
	rm -rf bab/*.pkl
	rm -rf dist

install:
	pip install -r requirements.txt
	python setup.py install
	pip install -r requirements_test.txt

lint:
	pytest bab --flake8
