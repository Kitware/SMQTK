CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
PYTHON = python2.7
.PHONY = test

all: svm-train svm-predict svm-scale svm-prob binary-train binary-predict

lib: svm.o
	$(CXX) -shared -dynamiclib svm.o -o libsvm.so.$(SHVER)

svm-test: svm-test.o svm.o
	$(CXX) svm-test.o svm.o -lgtest -lpthread -o svm-test

test: svm-test lib
	mkdir -p ./python/lib && cp libsvm.so.$(SHVER) python/lib/
	valgrind --leak-check=full ./svm-test
	cd python && $(PYTHON) svm_test.py

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm-prob: svm-prob.c
	$(CXX) $(CFLAGS) svm-prob.c -o svm-prob
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp

eval.o: eval.cpp eval.h svm.h
	$(CXX) $(CFLAGS) -c eval.cpp -o eval.o

svm-test.o: svm-test.cpp svm.o
	$(CXX) -std=c++11 $(CFLAGS) -c svm-test.cpp

binary-predict: binary-predict.c eval.o svm.o
	$(CXX) $(CFLAGS) binary-predict.c eval.o svm.o -o binary-predict -lm
binary-train: binary-train.c eval.o svm.o
	$(CXX) $(CFLAGS) binary-train.c eval.o svm.o -o binary-train -lm


clean:
	rm -f *~ svm.o svm-test.o eval.o svm-train svm-predict svm-scale \
	svm-prob binary-train binary-predict svm-test libsvm.so.$(SHVER) 
	rm -R ./python/lib
