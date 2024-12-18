build: FORCE
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install

test:
	python scripts/test.py

clean:
	pip uninstall custom_ops -y
	python setup.py clean

benchmark:
	python scripts/benchmark.py

count:
	python /home/wrt/code/pytorch/tools/kernel_count/kernel_count.py scripts/benchmark.py

.PHONY: FORCE
FORCE: