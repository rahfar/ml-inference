// Go inference server — ONNX Runtime backend.
//
// POST /predict  { "input": [[...150 floats...], ...] } → { "output": [[...30 floats...], ...] }
// GET  /health   → { "status": "ok" }
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	historySteps    = 30
	historyFeatures = 5
	futureSteps     = 15
	futureFeatures  = 2
	inputSize       = historySteps * historyFeatures // 150
	outputSize      = futureSteps * futureFeatures   // 30
)

type predictRequest struct {
	Input [][]float32 `json:"input"`
}

type predictResponse struct {
	Output [][]float32 `json:"output"`
}

// inferServer wraps a single ONNX session, serialised by a mutex so it is
// safe to share across concurrent HTTP handlers.
type inferServer struct {
	session *ort.DynamicAdvancedSession
	mu      sync.Mutex
}

func newInferServer(modelPath, libPath string) (*inferServer, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init ORT environment: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create ORT session: %w", err)
	}
	return &inferServer{session: session}, nil
}

func (s *inferServer) destroy() {
	s.session.Destroy()
	ort.DestroyEnvironment()
}

func (s *inferServer) predict(batch [][]float32) ([][]float32, error) {
	n := len(batch)
	flat := make([]float32, n*inputSize)
	for i, sample := range batch {
		if len(sample) != inputSize {
			return nil, fmt.Errorf("sample %d: got %d floats, want %d", i, len(sample), inputSize)
		}
		copy(flat[i*inputSize:], sample)
	}

	inputTensor, err := ort.NewTensor(
		ort.NewShape(int64(n), historySteps, historyFeatures),
		flat,
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	s.mu.Lock()
	outputs, err := s.session.Run([]ort.ArbitraryTensor{inputTensor})
	s.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("ORT run: %w", err)
	}
	defer func() {
		for _, t := range outputs {
			t.Destroy()
		}
	}()

	outTensor := outputs[0].(*ort.Tensor[float32])
	outData := outTensor.GetData()

	result := make([][]float32, n)
	for i := range n {
		row := make([]float32, outputSize)
		copy(row, outData[i*outputSize:])
		result[i] = row
	}
	return result, nil
}

func main() {
	port := flag.Int("port", 8000, "port to listen on")
	modelPath := flag.String("model", "/app/weights/model.onnx", "path to ONNX model")
	libPath := flag.String("lib", "/app/lib/libonnxruntime.so", "path to ORT shared library")
	flag.Parse()

	srv, err := newInferServer(*modelPath, *libPath)
	if err != nil {
		log.Fatalf("server init failed: %v", err)
	}
	defer srv.destroy()
	log.Printf("model loaded from %s", *modelPath)

	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /predict", func(w http.ResponseWriter, r *http.Request) {
		var req predictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Input) == 0 {
			http.Error(w, "empty input batch", http.StatusBadRequest)
			return
		}

		out, err := srv.predict(req.Input)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(predictResponse{Output: out})
	})

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
