build: FORCE
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install

test:
	rm -f test.txt
	python scripts/test.py

clean:
	pip uninstall custom_ops -y
	python setup.py clean

benchmark:
	rm -f output1.txt output2.txt output3.txt
	python scripts/benchmark.py
	diff output1.txt output2.txt

count:
	python /home/wrt/code/pytorch/tools/kernel_count/kernel_count.py scripts/benchmark.py

.PHONY: FORCE
FORCE: