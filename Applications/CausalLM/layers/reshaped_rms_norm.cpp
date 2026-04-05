// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   custom_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include "reshaped_rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  nntrainer::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::Initializer::ONES,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  unsigned int _from = from;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      ///@todo rms_norm_wrt_width_something() should be refactored to
      /// nntrainer::Tensor operation.
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();
  
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);
  
  ml::train::TensorDim in_dim = in.getDim();
  unsigned int b_size = in_dim.batch();
  unsigned int c_size = in_dim.channel();
  unsigned int h_size = in_dim.height();
  unsigned int w_size = in_dim.width();
  
  dgamma.setZero();
  
  // Reshape tensors for computation
  ml::train::TensorDim reshaped_dim(1, 1, h_size * (w_size / feature_size), feature_size);
  
  for (unsigned int b = 0; b < b_size; ++b) {
    for (unsigned int c = 0; c < c_size; ++c) {
      nntrainer::Tensor in_step =
        in.getSharedDataTensor(reshaped_dim, b * in_dim.getFeatureLen() + c * h_size * w_size, true);
      nntrainer::Tensor dy_step =
        dy.getSharedDataTensor(reshaped_dim, b * dy.getDim().getFeatureLen() + c * h_size * w_size, true);
      
      // Compute RMS for normalization
      unsigned int num_elements = feature_size;
      float *in_data = in_step.getData<float>();
      float *dy_data = dy_step.getData<float>();
      
      // dgamma = sum(dy * x * inv_rms, keepdims=True)
      // Compute inv_rms and accumulate gradient
      for (unsigned int i = 0; i < in_step.getDim().height(); ++i) {
        // Compute mean of squares
        float mean_sq = 0.0f;
        for (unsigned int j = 0; j < num_elements; ++j) {
          float val = in_data[i * num_elements + j];
          mean_sq += val * val;
        }
        mean_sq /= num_elements;
        
        float inv_rms = 1.0f / sqrtf(mean_sq + epsilon);
        
        // Accumulate gradient for gamma
        float *dgamma_data = dgamma.getData<float>();
        for (unsigned int j = 0; j < num_elements; ++j) {
          dgamma_data[j] += dy_data[i * num_elements + j] * in_data[i * num_elements + j] * inv_rms;
        }
      }
    }
  }
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();
  
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  
  ml::train::TensorDim in_dim = in.getDim();
  unsigned int b_size = in_dim.batch();
  unsigned int c_size = in_dim.channel();
  unsigned int h_size = in_dim.height();
  unsigned int w_size = in_dim.width();
  
  // Reshape tensors for computation
  ml::train::TensorDim reshaped_dim(1, 1, h_size * (w_size / feature_size), feature_size);
  
  for (unsigned int b = 0; b < b_size; ++b) {
    for (unsigned int c = 0; c < c_size; ++c) {
      nntrainer::Tensor in_step =
        in.getSharedDataTensor(reshaped_dim, b * in_dim.getFeatureLen() + c * h_size * w_size, true);
      nntrainer::Tensor dy_step =
        dy.getSharedDataTensor(reshaped_dim, b * dy.getDim().getFeatureLen() + c * h_size * w_size, true);
      nntrainer::Tensor dx_step =
        dx.getSharedDataTensor(reshaped_dim, b * dx.getDim().getFeatureLen() + c * h_size * w_size, true);
      
      unsigned int num_elements = feature_size;
      float *in_data = in_step.getData<float>();
      float *dy_data = dy_step.getData<float>();
      float *dx_data = dx_step.getData<float>();
      float *gamma_data = gamma.getData<float>();
      
      // Backward pass
      for (unsigned int i = 0; i < in_step.getDim().height(); ++i) {
        // Compute mean of squares (same as forward)
        float mean_sq = 0.0f;
        for (unsigned int j = 0; j < num_elements; ++j) {
          float val = in_data[i * num_elements + j];
          mean_sq += val * val;
        }
        mean_sq /= num_elements;
        
        float inv_rms = 1.0f / sqrtf(mean_sq + epsilon);
        float inv_rms_sq = inv_rms * inv_rms;
        
        // Compute c = mean(gamma * dy * x)
        float c = 0.0f;
        for (unsigned int j = 0; j < num_elements; ++j) {
          c += gamma_data[j] * dy_data[i * num_elements + j] * in_data[i * num_elements + j];
        }
        c /= num_elements;
        
        // dx = inv_rms * (gamma * dy - x * c * inv_rms)
        for (unsigned int j = 0; j < num_elements; ++j) {
          float gamma_dy = gamma_data[j] * dy_data[i * num_elements + j];
          dx_data[i * num_elements + j] = inv_rms * (gamma_dy - in_data[i * num_elements + j] * c * inv_rms_sq);
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
