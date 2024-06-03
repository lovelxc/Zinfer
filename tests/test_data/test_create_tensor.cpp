// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 23-1-15.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/tensor.hpp"

TEST(TestCreateTensor, TensorInit1D) {
  using zinfer::Tensor;
  Tensor<float> f1(4);
  const auto& raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  EXPECT_EQ(raw_shapes.size(), 1);
  f1.randn();
  const uint32_t size = f1.size();
  EXPECT_EQ(size, 4);
  f1.print();
}

TEST(TestCreateTensor, TensorInit2D) {
  using zinfer::Tensor;
  Tensor<float> f1(4, 3);
  const auto& raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  EXPECT_EQ(raw_shapes.size(), 2);
  f1.randn();
  const uint32_t size = f1.size();
  EXPECT_EQ(size, 12);
  f1.print();
}

TEST(TestCreateTensor, TensorInit3D) {
  using zinfer::Tensor;
  Tensor<float> f1(4, 3, 2);
  const auto& raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D-----------------------";
  EXPECT_EQ(raw_shapes.size(), 3);
  f1.randn();
  const uint32_t size = f1.size();
  EXPECT_EQ(size, 24);
  f1.print();
}
TEST(TestCreateTensor, TensorFills1D) {
  using zinfer::Tensor;
  Tensor<float> f1(4);
  const int size = 4;
  std::vector<float> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = i + 1;
  }
  f1.fills(vec, true);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(f1.index(i), vec[i]);
  }
}
TEST(TestCreateTensor, TensorFills2D) {
  using zinfer::Tensor;
  Tensor<float> f1(4, 5);
  const int size = 4 * 5;
  std::vector<float> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = i + 1;
  }
  f1.fills(vec, true);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(f1.index(i, true), vec[i]);
  }
}
TEST(TestCreateTensor, TensorFills3D) {
  using zinfer::Tensor;
  Tensor<float> f1(4, 3, 2);
  const int size = 4 * 3 * 2;
  std::vector<float> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = i + 1;
  }
  f1.fills(vec, true);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(f1.index(i, true), vec[i]);
  }
}
