build:
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install

test:
	python test.py