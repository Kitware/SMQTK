CXX? = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 1

all: svm-train svm-predict svm-scale

lib: svm.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o eval.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o eval.o -o svm-predict -lm
svm-train: svm-train.c svm.o eval.o
	$(CXX) $(CFLAGS) svm-train.c svm.o eval.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
eval.o: eval.cpp eval.h svm.h
	$(CXX) $(CFLAGS) -c eval.cpp -o eval.o

clean:
	rm -f *~ svm.o eval.o svm-train svm-predict svm-scale
