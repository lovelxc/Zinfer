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

TEST(TestGetValues, TensorIntValues1) {
  using zinfer::Tensor;
  Tensor<int> f1(2, 3, 4);
  f1.randn(0, +100);
}

TEST(TestGetValues, TensorFloatValues1) {
  using zinfer::Tensor;
  Tensor<float> f1(2, 3, 4);
  f1.randn(0, +100);
}

TEST(TestGetValues, TensorIntFill1) {
  using zinfer::Tensor;
  Tensor<int> f1(2, 3, 4);
  int v = 23;
  f1.fill(v);
  for (auto i = 0; i < f1.size(); i++) {
    EXPECT_EQ(f1.index(i, true), v);
  }
}

TEST(TestGetValues, TensorFloatFill1) {
  using zinfer::Tensor;
  Tensor<float> f1(2, 3, 4);
  float v = 2.0;
  f1.fill(v);
  for (auto i = 0; i < f1.size(); i++) {
    EXPECT_EQ(f1.index(i, true), v);
  }
}