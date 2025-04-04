build_msm:
	mkdir -p work
	nvcc -o work/test_msm -std=c++17 -arch=sm_80 -I. -I../../include -I../../../../include tests/msm_test.cu

test_msm: build_msm	
	work/test_msm

build_huge_msm:
	mkdir -p work
	nvcc -o work/test_huge_msm -std=c++17 -arch=sm_80 -I. -I../../include -I../../../../include tests/huge_msm_test.cu

test_huge_msm: build_huge_msm	
	work/test_huge_msm