#########################################################################
#                                                                       #
#     COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen      #
#                                                                       #
#########################################################################

#########################################################################
# set dependency file:                                                  #
#########################################################################
INSTDIR = $(shell pwd)
LOGFILE =

#########################################################################
# define targets:                                                       #
#########################################################################
TARGETS		= asrlab

#########################################################################
# compiler options                                                      #
#########################################################################
CXX       = g++
CXXFLAGS += -Wall -O2 --std=c++11 -pedantic -msse -msse2 -msse3 -flto
DFLAGS    = -g #-fsanitize=address -fno-omit-frame-pointer #-fno-inline
IFLAGS    = -Irapidjson/include/
LIBS      = -lm -lrt -lgsl -lblas
OBJDIR    = .build
SOURCES   = $(wildcard *.cpp)
HEADERS   = $(wildcard *.hpp)

#########################################################################
# all:                                                                  #
#########################################################################
all: $(TARGETS)

#########################################################################
# asr_engine:                                                           #
#########################################################################
ASRLAB_O	= $(OBJDIR)/main.o\
						$(OBJDIR)/Alignment.o\
						$(OBJDIR)/Config.o\
						$(OBJDIR)/Corpus.o\
						$(OBJDIR)/FeedForwardLayer.o\
						$(OBJDIR)/IO.o\
						$(OBJDIR)/Lexicon.o\
						$(OBJDIR)/Mixtures.o\
						$(OBJDIR)/NetworkLayer.o\
						$(OBJDIR)/NeuralNetwork.o\
						$(OBJDIR)/NNTraining.o\
						$(OBJDIR)/OutputLayer.o\
						$(OBJDIR)/Recognizer.o\
						$(OBJDIR)/SignalAnalysis.o\
						$(OBJDIR)/TdpModel.o\
						$(OBJDIR)/Training.o\
						$(OBJDIR)/Util.o

asrlab:	$(ASRLAB_O)
		$(CXX) $(LDFLAGS) -o ../asrlab $(ASRLAB_O) $(DFLAGS) $(LIBS)

#########################################################################
# clean / clean_all:                                                    #
#########################################################################
.PHONY: clean clean_all

clean:
	rm -f core core.* *~ *.il .build/*.o

clean_all: clean
	rm asrlab

#########################################################################
# generate dependency file if necessary                                 #
#########################################################################
$(OBJDIR)/%.o: %.cpp $(HEADERS)
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) $(IFLAGS) $(DFLAGS) $< -o $@
