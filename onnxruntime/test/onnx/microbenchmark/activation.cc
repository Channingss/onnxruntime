#include "core/session/ort_env.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/data_transfer_manager.h"
#include "core/util/thread_utils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/execution_frame.h"
#include "contrib_ops/cpu/activations.h"
#include "core/providers/cpu/activation/activations.h"
#include <onnx/defs/attr_proto_util.h>
#include <benchmark/benchmark.h>
#include <random>

using namespace onnxruntime;
using namespace onnx;
extern OrtEnv* env;

static float* GenerateFloatArray(size_t batch_size, float low, float high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(low, high);
  float* data = (float*)_aligned_malloc(sizeof(float) * batch_size, 64);
  for (size_t i = 0; i != batch_size; ++i) {
    data[i] = dist(gen);
  }
  return data;
}

class Allocs : public IExecutionProvider {
 private:
  std::shared_ptr<CPUAllocator> alloc = std::make_shared<CPUAllocator>();

 public:
  Allocs() : IExecutionProvider("fake"){};
  virtual AllocatorPtr GetAllocator(int, OrtMemType) const {
    return alloc;
  }
};

struct KernelAndDef {
  std::unique_ptr<KernelDef> def;
  std::unique_ptr<Model> model;
  std::unique_ptr<logging::Logger> test_logger;
  std::unique_ptr<OpKernel> kernel;
  std::unique_ptr<OrtValueNameIdxMap> ort_value_idx_map = std::make_unique<OrtValueNameIdxMap>();
  std::unique_ptr<Allocs> a = std::make_unique<Allocs>();

  template <typename KernelType>
  static KernelAndDef CreateKernel(const std::string& op_name, const std::string& domain,
                                   const std::vector<AttributeProto>& attrs, int64_t batch_size) {
    std::unordered_map<std::string, int> domain2Version;
    domain2Version[""] = 12;
    domain2Version[kMSDomain] = 1;
    KernelAndDef out;
    out.test_logger = env->GetLoggingManager()->CreateLogger("test");
    out.model = std::make_unique<Model>("graph_1", false, *out.test_logger);
    auto& graph = out.model->MainGraph();
    TypeProto tensor_float;
    tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
    auto& input_arg = graph.GetOrCreateNodeArg("input", &tensor_float);
    auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_float);
    out.ort_value_idx_map->Add("input");
    out.ort_value_idx_map->Add("output");
    std::unordered_map<std::string, AttributeProto> attributes;
    for (const AttributeProto& p : attrs) {
      attributes[p.name()] = p;
    }
    Node& main_node = graph.AddNode("main", op_name, "", {&input_arg}, {&output_arg}, &attributes, domain);
    ORT_THROW_IF_ERROR(graph.Resolve());
    main_node.SetExecutionProviderType("fake");
    out.def = KernelDefBuilder()
                  .SetName(op_name)
                  .SetDomain(domain)
                  .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                  .Build();
    OpKernelInfo info(main_node, *out.def, *out.a, {}, {}, {}, {});
    out.kernel = std::make_unique<KernelType>(info);
    return out;
  }
};

class MyIExecutionFrame : public IExecutionFrame {
 private:
  IExecutionProvider& a_;

 public:
  MyIExecutionFrame(IExecutionProvider& a, const std::vector<int>& feed_mlvalue_idxs,
                    const std::vector<OrtValue>& feeds, const std::unordered_map<int, OrtValue>& initializers,
                    const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                    const OrtValueNameIdxMap& ort_value_idx_map, const NodeIndexInfo& node_index_info)
      : IExecutionFrame(feed_mlvalue_idxs, feeds, initializers, fetch_mlvalue_idxs, fetches, ort_value_idx_map,
                        node_index_info),
        a_(a) {
  }

  AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo& info) const {
    return a_.GetAllocator(info.id, info.mem_type);
  }

  Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_index, const TensorShape* shape, size_t) {
    using T = float;
    if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
      return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
    }
    size_t size;
    int64_t len = shape->Size();
    if (len < 0) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
    }
    if (static_cast<uint64_t>(len) > std::numeric_limits<size_t>::max()) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape is too large");
    }

    if (!IAllocator::CalcMemSizeForArrayWithAlignment<0>(static_cast<size_t>(len), sizeof(T), &size)) {
      return Status(ONNXRUNTIME, FAIL, "size overflow");
    }
    auto alloc = a_.GetAllocator(0, OrtMemTypeDefault);
    std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<T>(), *shape, alloc);

    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
    return Status::OK();
  }
};

template <typename KernelType>
static void RunSingleNode(const std::string& op_name, const std::string& domain,
                          const std::vector<AttributeProto>& attrs, benchmark::State& state, float low = -1.0f,
                          float high = 1.0f) {
  const int64_t batch_size = state.range(0);
  float* output = (float*)_aligned_malloc(sizeof(float) * static_cast<size_t>(batch_size), 64);
  float* data = GenerateFloatArray(batch_size, low, high);
  KernelAndDef k = KernelAndDef::CreateKernel<KernelType>(op_name, domain, attrs, batch_size);

  std::vector<int> feed_mlvalue_idxs(1);
  std::vector<int> fetch_mlvalue_idxs(1);
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("input", feed_mlvalue_idxs[0]));
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("output", fetch_mlvalue_idxs[0]));

  std::vector<OrtValue> feeds(1);
  std::vector<OrtValue> fetches(1);
  std::vector<int64_t> shapes(static_cast<size_t>(1), batch_size);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  OrtMemoryInfo info("cpu", OrtDeviceAllocator);
  feeds[0].Init(new Tensor(DataTypeImpl::GetType<float>(), shapes, data, info), ml_tensor, ml_tensor->GetDeleteFunc());
  fetches[0].Init(new Tensor(DataTypeImpl::GetType<float>(), shapes, output, info), ml_tensor,
                  ml_tensor->GetDeleteFunc());
  GraphViewer v(k.model->MainGraph());
  NodeIndexInfo node_index_info(v, *k.ort_value_idx_map);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP, nullptr));
  MyIExecutionFrame f(*k.a, feed_mlvalue_idxs, feeds, {}, fetch_mlvalue_idxs, fetches, *k.ort_value_idx_map,
                      node_index_info);
  for (auto _ : state) {
    OpKernelContext c(&f, k.kernel.get(), tp.get(), *k.test_logger);
    Status st = k.kernel->Compute(&c);
    if (!st.IsOK())
      state.SkipWithError(st.ErrorMessage().c_str());
  }
  _aligned_free(data);
  _aligned_free(output);
}

static void BM_GeluCompute(benchmark::State& state) {
  RunSingleNode<contrib::Gelu<float>>("Gelu", kMSDomain, {}, state);
}

BENCHMARK(BM_GeluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_ScaledTanhCompute(benchmark::State& state) {
  RunSingleNode<contrib::ScaledTanh<float>>("ScaledTanh", kMSDomain,
                                            {MakeAttribute("alpha", 0.8f), MakeAttribute("beta", 0.3f)}, state);
}

BENCHMARK(BM_ScaledTanhCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

static void BM_EluCompute(benchmark::State& state) {
  RunSingleNode<Elu<float>>("Elu", "",
                            {
                                MakeAttribute("alpha", 0.8f),
                            },
                            state);
}

BENCHMARK(BM_EluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

static void BM_HardSigmoidCompute(benchmark::State& state) {
  RunSingleNode<HardSigmoid<float>>("HardSigmoid", "", {MakeAttribute("alpha", 0.2f), MakeAttribute("beta", 0.5f)},
                                    state, 0.1f, 0.6f);
}

BENCHMARK(BM_HardSigmoidCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_LeakyReluCompute(benchmark::State& state) {
  RunSingleNode<LeakyRelu<float>>("LeakyRelu", "", {MakeAttribute("alpha", 0.2f)}, state);
}

BENCHMARK(BM_LeakyReluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000);

static void BM_ParametricSoftplusCompute(benchmark::State& state) {
  RunSingleNode<ParametricSoftplus<float>>("Softplus", "", {MakeAttribute("alpha", 1.0f), MakeAttribute("beta", 1.0f)},
                                           state, -2.0f, 2.0f);
}

BENCHMARK(BM_ParametricSoftplusCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Selu(benchmark::State& state) {
  RunSingleNode<Selu<float>>("Selu", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Selu)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);