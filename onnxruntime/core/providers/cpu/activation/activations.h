// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/threadpool.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
using concurrency::ThreadPool;

#define EIGEN_X ConstEigenVectorArrayMap<T>(X->template Data<T>(), X->Shape().Size())
#define EIGEN_X_VAR(var) ConstEigenVectorArrayMap<T> var(X->template Data<T>(), X->Shape().Size())
#define EIGEN_Y EigenVectorArrayMap<T>(Y->template MutableData<T>(), Y->Shape().Size())
#define EIGEN_Y_VAR(var) EigenVectorArrayMap<T> var(Y->template MutableData<T>(), Y->Shape().Size())

template <typename T>
class Elu final : public OpKernel {
 public:
  explicit Elu(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("alpha", &alpha_));
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    float alpha = alpha_;
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 40.0;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [alpha, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (xm >= 0).select(xm, alpha * (xm.exp() - 1));
    });
    return Status::OK();
  }

 private:
  float alpha_;
};

template <typename T>
class HardSigmoid final : public OpKernel {
 public:
  explicit HardSigmoid(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("alpha", &alpha_));
    ORT_THROW_IF_ERROR(info.GetAttr("beta", &beta_));
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    float alpha = alpha_;
    float beta = beta_;
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 0.5;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [alpha, beta, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (((T)alpha * xm + (T)beta).cwiseMin(1.0f)).cwiseMax(0.0f);
    });
    return Status::OK();
  }

 private:
  float alpha_;
  float beta_;
};

template <typename T>
class LeakyRelu final : public OpKernel {
 public:
  explicit LeakyRelu(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("alpha", &alpha_));
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    float alpha = alpha_;
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 25;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [alpha, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (xm >= 0).select(xm, (T)alpha * xm);
    });
    return Status::OK();
  }

 private:
  float alpha_;
};

// This kernel is used by multiple ops so the attributes may not exists
template <typename T>
class ParametricSoftplus final : public OpKernel {
 public:
  explicit ParametricSoftplus(const OpKernelInfo& info)
      : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)), beta_(info.GetAttrOrDefault("beta", 1.0f)) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    float alpha = alpha_;
    float beta = beta_;
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 15;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [alpha, beta, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (T)alpha *
           (xm * (T)beta > 0)
               .select(xm * (T)beta + ((-xm * (T)beta).exp() + 1.0f).log(), ((xm * (T)beta).exp() + 1.0f).log());
    });
    return Status::OK();
  }

 private:
  const float alpha_;
  const float beta_;
};

template <typename T>
class Relu : public OpKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = EIGEN_X.cwiseMax(0);
    return Status::OK();
  }
};

template <typename T>
class Selu final : public OpKernel {
 public:
  explicit Selu(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(info.GetAttr("alpha", &alpha_));
    ORT_THROW_IF_ERROR(info.GetAttr("gamma", &gamma_));
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    float alpha = alpha_;
    float gamma = gamma_;
    ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input_size);
    // The cost comes from microbenchmark(manual tunning).
    const double cost = 4;
    const T* data = X->template Data<T>();
    T* output = Y->template MutableData<T>();
    ThreadPool::TryParallelFor(tp, batch_size, cost, [alpha, gamma, data, output](ptrdiff_t first, ptrdiff_t last) {
      ptrdiff_t len = last - first;
      T* output_ptr = output + first;
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (T)gamma * (xm.cwiseMax(0.0f) + ((T)alpha * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
    });
    return Status::OK();
  }

 private:
  float alpha_;
  float gamma_;
};

template <typename T>
class Sigmoid final : public OpKernel {
 public:
  explicit Sigmoid(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
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
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
    });
    return Status::OK();
  }
};

template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const;

template <typename T>
class Softsign final : public OpKernel {
 public:
  explicit Softsign(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
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
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = (1 + xm.abs()).inverse() * xm;
    });
    return Status::OK();
  }
};

template <typename T>
class Tanh final : public OpKernel {
 public:
  Tanh(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
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
      onnxruntime::ConstEigenVectorArrayMap<T> xm(data + first, len);
      onnxruntime::EigenVectorArrayMap<T> ym(output_ptr, len);
      ym = xm.tanh();
    });
    return Status::OK();
  }
};

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const;

template <typename T>
class ThresholdedRelu final : public OpKernel {
 public:
  ThresholdedRelu(const OpKernelInfo& info) : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)) {
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm > (T)alpha_).select(xm, 0);
    return Status::OK();
  }

 private:
  const float alpha_;
};
}  // namespace onnxruntime
