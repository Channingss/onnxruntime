// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion)                              \
  ONNX_CPU_OPERATOR_KERNEL(                                                                          \
      alias,                                                                                         \
      sinceVersion,                                                                                  \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) \
  REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
// SoftPlus is the default case for ParametricSoftPlus
REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(Softplus, ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const {
  using T = float;
  const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape()); 
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 1;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      MlasComputeLogistic(data + first, output_ptr, static_cast<size_t>(len));
    });    
    return Status::OK();  
}

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const {
    using T = float;
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape()); 
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 2;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      MlasComputeTanh(data + first, output_ptr, static_cast<size_t>(len));
    });    
    return Status::OK();   
}
}  // namespace onnxruntime
