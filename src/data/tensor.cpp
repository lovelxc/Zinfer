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

// Created by fss on 22-11-12.

#include <cstdint>
#include <random>
#include "data/tensor.hpp"
namespace zinfer {

template <typename T>
Tensor<T>::Tensor(uint32_t size) {
  this->raw_shapes_ = {size};
  this->data_ = arma::Cube<T>(1, size, 1);
}

template <typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
  this->data_ = arma::Cube<T>(rows, cols, 1);
  if (rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }
}

template <typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  this->data_ = arma::Cube<T>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  this->raw_shapes_ = std::vector<uint32_t>(3, 1);
  std::copy(shapes.begin(), shapes.end(), this->raw_shapes_.begin() + remaining);

  uint32_t channels = this->raw_shapes_.at(0);
  uint32_t rows = this->raw_shapes_.at(1);
  uint32_t cols = this->raw_shapes_.at(2);

  this->data_ = arma::Cube<T>(rows, cols, channels);
  if (channels == 1)
    this->raw_shapes_.erase(this->raw_shapes_.begin());
  if (rows == 1)
    this->raw_shapes_.erase(this->raw_shapes_.begin());
}

template <typename T>
std::vector<uint32_t> Tensor<T>::shapes() const {
  CHECK(this->data_.size());
  return {this->channels(), this->rows(), this->cols()};
}

template <typename T>
const std::vector<uint32_t>& Tensor<T>::raw_shapes() const {
  CHECK(this->data_.size());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

template <typename T>
const T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_(row, col, channel);
}

template <typename T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_(row, col, channel);
}

template <typename T>
const T& Tensor<T>::index(uint32_t offset, bool as_raw_major) const {
  CHECK_LT(offset, this->size());
  uint32_t index = offset;
  if (as_raw_major && this->dim() > 1) {
    uint32_t channel = offset / this->plane_size();
    uint32_t remaining = offset % this->plane_size();
    uint32_t row = remaining / this->cols();
    uint32_t col = remaining % this->cols();

    return this->at(channel, row, col);
  }
  return this->data_.at(index);
}

template <typename T>
T& Tensor<T>::index(uint32_t offset, bool as_raw_major) {
  CHECK_LT(offset, this->size());
  uint32_t index = offset;
  if (as_raw_major && this->dim() > 1) {
    uint32_t channel = offset / this->plane_size();
    uint32_t remaining = offset % this->plane_size();
    uint32_t row = remaining / this->cols();
    uint32_t col = remaining % this->cols();

    return this->at(channel, row, col);
  }
  return this->data_.at(index);
}

template <typename T>
uint32_t Tensor<T>::rows() const {
  CHECK(this->data_.size()) << "The data area of the tensor is empty.";
  return this->data_.n_rows;
}

template <typename T>
uint32_t Tensor<T>::cols() const {
  CHECK(this->data_.size()) << "The data area of the tensor is empty.";
  return this->data_.n_cols;
}

template <typename T>
uint32_t Tensor<T>::channels() const {
  CHECK(this->data_.size()) << "The data area of the tensor is empty.";
  return this->data_.n_slices;
}
template <typename T>
uint32_t Tensor<T>::dim() const {
  return this->raw_shapes_.size();
}

template <typename T>
size_t Tensor<T>::size() const {
  CHECK(this->data_.size()) << "The data area of the tensor is empty.";
  return this->data_.size();
}

template <typename T>
size_t Tensor<T>::plane_size() const {
  CHECK(this->data_.size()) << "The data area of the tensor is empty.";
  return this->rows() * this->cols();
}
template <typename T>
const arma::Mat<T>& Tensor<T>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

template <typename T>
void Tensor<T>::randn(T mean, T standard_deviation) {
  arma::arma_rng::set_seed_random();
  this->data_ =
      std::move(arma::randn<arma::Cube<T>>(this->rows(), this->cols(), this->channels(),
                                           arma::distr_param(mean, standard_deviation)));
}

template <typename T>
void Tensor<T>::randu(T min, T max) {
  arma::arma_rng::set_seed_random();
  this->data_ = std::move(arma::randu<arma::Cube<T>>(
      this->rows(), this->cols(), this->channels(), arma::distr_param(min, max)));
}

template <typename T>
void Tensor<T>::print() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    std::ostringstream oss;
    this->data_.slice(i).brief_print(oss);
    LOG(INFO) << "\n" << oss.str();
  }
}

template <typename T>
void Tensor<T>::ones() {
  this->data_.ones();
}

template <typename T>
void Tensor<T>::zeros() {
  this->data_.zeros();
}

template <typename T>
void Tensor<T>::fill(T value) {
  this->data_.fill(value);
}

template <typename T>
void Tensor<T>::fills(std::vector<T> v, bool row_major) {
  CHECK_EQ(this->size(), v.size()) << "vector size isn't match";
  if (row_major) {
    uint32_t planes = this->plane_size();
    for (auto i = 0; i < this->channels(); ++i) {
      auto& channel_data = this->data_.slice(i);
      const arma::Mat<T>& new_mat =
          arma::Mat<T>(v.data() + i * planes, this->cols(), this->rows());
      ;
      channel_data = std::move(new_mat.t());
    }
  } else {
    std::copy(v.begin(), v.end(), this->data_.memptr());
  }
}

template class Tensor<float>;
template class Tensor<int32_t>;
template class Tensor<uint8_t>;
}  // namespace zinfer