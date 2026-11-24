# The C++ server. Needs: Homebrew's drogon and simdjson, torch from the
# project's venv (uv sync), and the CatBoost model library (downloaded here).
#
#   brew install drogon simdjson
#   make -C cpp          # or `make cpp` from the repo root

CATBOOST := 1.2.10
TORCH    := $(shell cd .. && uv run --quiet python -c "import os, torch; print(os.path.dirname(torch.__file__))")
BREW     := $(shell brew --prefix)

CXXFLAGS := -std=c++20 -O3 -mcpu=native -flto -DNDEBUG -Wall -Wextra
INCLUDES := -Ithird_party -isystem $(BREW)/include \
            -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LDFLAGS  := -L$(BREW)/lib -L$(TORCH)/lib -Lthird_party \
            -Wl,-rpath,$(BREW)/lib -Wl,-rpath,$(TORCH)/lib -Wl,-rpath,$(abspath third_party)
LDLIBS   := -ldrogon -ltrantor -ljsoncpp -lsimdjson -ltorch -ltorch_cpu -lc10 -lcatboostmodel

RELEASE  := https://github.com/catboost/catboost/releases/download/v$(CATBOOST)
SOURCE   := https://raw.githubusercontent.com/catboost/catboost/v$(CATBOOST)

all: build/server ../models/weights/lstm.ts

build/server: server.cpp third_party/c_api.h third_party/libcatboostmodel.dylib
	@test -f $(BREW)/include/drogon/drogon.h || { echo "Drogon not found: brew install drogon"; exit 1; }
	@test -f $(BREW)/include/simdjson.h || { echo "simdjson not found: brew install simdjson"; exit 1; }
	@mkdir -p build
	clang++ $(CXXFLAGS) $(INCLUDES) server.cpp $(LDFLAGS) $(LDLIBS) -o $@

third_party/libcatboostmodel.dylib:
	@mkdir -p third_party
	curl -sSfL -o $@ $(RELEASE)/libcatboostmodel-darwin-universal2-$(CATBOOST).dylib

third_party/c_api.h:
	@mkdir -p third_party
	curl -sSfL -o $@ $(SOURCE)/catboost/libs/model_interface/c_api.h

../models/weights/lstm.ts: export_lstm.py ../models/weights/lstm.pt
	cd .. && uv run python cpp/export_lstm.py

clean:
	rm -rf build

.PHONY: all clean
