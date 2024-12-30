build: FORCE
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install -- -j 8



benchmark_mm:
	python scripts/benchmark_mm.py

clean:
	pip uninstall custom_ops -y
	python setup.py clean

benchmark_trans:
	python src/transformer/benchmark.py

benchmark_log:
	rm -f output1.txt output2.txt LinearLog.txt
	python scripts/benchmark.py
	diff output1.txt output2.txt

# count:
# 	python /home/wrt/code/pytorch/tools/kernel_count/kernel_count.py scripts/benchmark.py

.PHONY: FORCE
FORCE: