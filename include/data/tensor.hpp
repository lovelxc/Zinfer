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

#ifndef ZINFER_DATA_BLOB_HPP_
#define ZINFER_DATA_BLOB_HPP_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>
namespace zinfer {
template <typename T>
class Tensor {
 public:
  /**
   * @brief Construct a new empty Tensor
   */
  explicit Tensor() = default;

  /**
   * @brief Construct a 3D Tensor
   *
   * @param channels Number of channels
   * @param rows Number of rows
   * @param cols Number of columns
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /**
   * @brief Construct a 1D Tensor
   *
   * @param size Vector size
   */
  explicit Tensor(uint32_t size);

  /**
   * @brief Construct a 2D matrix Tensor
   *
   * @param rows Number of rows
   * @param cols Number of columns
   */
  explicit Tensor(uint32_t rows, uint32_t cols);

  /**
   * @brief Construct Tensor with shape
   *
   * @param shapes Tensor dimensions
   */
  explicit Tensor(const std::vector<uint32_t>& shapes);

  /**
   * @brief Gets number of rows
   *
   * @return Number of rows
   */
  uint32_t rows() const;

  /**
   * @brief Gets number of columns
   *
   * @return Number of columns
   */
  uint32_t cols() const;

  /**
   * @brief Gets number of channels
   *
   * @return Number of channels
   */
  uint32_t channels() const;

  /**
   * @brief Gets dimension of tensor
   *
   * @return Number of dimension
   */
  uint32_t dim() const;

  /**
   * @brief Sets the tensor data
   *
   * @param data Data to set
   */
  void set_data(const arma::Cube<T>& data);

  /**
   * @brief Gets tensor shape
   *
   * @return Tensor dimensions
   */
  std::vector<uint32_t> shapes() const;

  /**
   * @brief Gets raw tensor shape
   *
   * @return Raw tensor dimensions
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * @brief Gets const element at location
   *
   * @param channel Channel
   * @param row Row index
   * @param col Column index
   * @return Element value
   */
  const T& at(uint32_t channel, uint32_t row, uint32_t col) const;

  /**
   * @brief Gets element reference at location
   *
   * @param channel Channel
   * @param row Row index
   * @param col Column index
   * @return Element reference
   */
  T& at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * @brief Gets element reference at offset
   *
   * @param offset Element offset
   * @param as_raw_major access whether in raw_major method
   * @return Element reference
   */
  T& index(uint32_t offset, bool as_raw_major = false);

  /**
   * @brief Gets element at offset
   *
   * @param offset Element offset
   * @param as_raw_major access whether in raw_major method
   * @return Element value
   */
  const T& index(uint32_t offset, bool as_raw_major) const;

  /**
   * @brief Gets tensor size
   *
   * @return Tensor size
   */
  size_t size() const;
  /**
   * @brief Gets tensor plane size
   *
   * @return Tensor plane size
   */
  size_t plane_size() const;
  /**
   * @brief Gets channel matrix
   *
   * @param channel Channel index
   * @return Channel matrix
   */
  const arma::Mat<T>& slice(uint32_t channel) const;

  /**
   * @brief Initializes with normal distribution
   *
   * @param mean Mean value
   * @param var Variance
   */
  void randn(T mean = 0, T standard_deviation = 1);

  /**
   * @brief Initializes with uniform distribution
   *
   * @param min Minimum value
   * @param max Maximum value
   */
  void randu(T min = -1, T max = 1);

  /**
   * @brief Prints tensor
   */
  void print();

  /**
   * @brief Sets all elem with one
   */
  void ones();

  /**
   * @brief Sets all elem with zero
   */
  void zeros();

  /**
   * @brief Fills tensor with value
   *
   * @param value Fill value
   */
  void fill(T value);

  /**
   * @brief Fills tensor with a vector
   *
   * @param value Fill vector
   * @param row_major is row_major
   */
  void fills(std::vector<T> v, bool row_major);

  /**
   * @brief Reshape tensor
   *
   * @param shapes New shape
   * @param row_major Row-major or column-major
   */
  void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

 private:
  /// Raw tensor dimensions
  std::vector<uint32_t> raw_shapes_;

  /// Tensor data
  arma::Cube<T> data_;
};

}  // namespace zinfer

#endif  // ZINFER_DATA_BLOB_HPP_