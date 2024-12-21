build: FORCE
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install

test:
	rm -f test.txt
	python scripts/test_valid.py

time:
	python scripts/test_timecost.py

clean:
	pip uninstall custom_ops -y
	python setup.py clean

benchmark:
	python src/transformer/benchmark.py

benchmark_log:
	rm -f output1.txt output2.txt LinearLog.txt
	python scripts/benchmark.py
	diff output1.txt output2.txt

# count:
# 	python /home/wrt/code/pytorch/tools/kernel_count/kernel_count.py scripts/benchmark.py

.PHONY: FORCE
FORCE: