build: FORCE
	pip uninstall custom_ops -y
	python setup.py clean
	python setup.py install

benchmark_mm:
	@echo "Testing MatMul Kernel. May take 30 seconds..."
	@echo "---------------------------------------------"
	@python scripts/benchmark_mm.py
	@echo "---------------------------------------------"

benchmark_softmax:
	@echo "Testing Softmax Kernel. May take 10 seconds..."
	@echo "---------------------------------------------"
	@python scripts/benchmark_softmax.py
	@echo "---------------------------------------------"

benchmark_transpose:
	@echo "Testing Transpose Kernel. May take 20 seconds..."
	@echo "---------------------------------------------"
	@python scripts/benchmark_transpose.py
	@echo "---------------------------------------------"

benchmark_trans:
	@echo "Testing Custom Transformer. May take 20 seconds..."
	@echo "---------------------------------------------"
	@python src/transformer/benchmark.py
	@echo "---------------------------------------------"

benchmark_all: benchmark_mm benchmark_softmax benchmark_transpose benchmark_trans
	@echo "All benchmarks done."

clean:
	pip uninstall custom_ops -y
	python setup.py clean

.PHONY: FORCE
FORCE: