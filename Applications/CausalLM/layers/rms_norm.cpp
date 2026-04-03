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
#include <iostream>

#include "rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;



void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);

  // Cache inv_rms for backward pass: shape (batch, channel, height, 1)
  nntrainer::TensorDim inv_rms_dim(
    dim[0].batch(), dim[0].channel(), dim[0].height(), 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::inv_rms] = context.requestTensor(
    inv_rms_dim, "inv_rms", nntrainer::props::InitializerInfo::Enum::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  // Temp tensor for calcDerivative: same shape as input
  wt_idx[RMSParams::temp_full] = context.requestTensor(
    dim[0], "temp_full", nntrainer::props::InitializerInfo::Enum::NONE, false,
    nntrainer::TensorLifespan::CALC_DERIV_LIFESPAN);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &inv_rms = context.getTensor(wt_idx[RMSParams::inv_rms]);

  // inv_rms = 1 / sqrt(mean(x^2) + eps)
  // average along width axis (axis 3)
  in.multiply(in, out);         // out = x^2 (temp use)
  out.average(3, inv_rms);      // inv_rms = mean(x^2)
  inv_rms.add_i(epsilon);       // inv_rms = mean(x^2) + eps
  inv_rms.inv_sqrt_i();         // inv_rms = 1/sqrt(mean(x^2) + eps)

  // out = x * inv_rms * gamma
  in.multiply(inv_rms, out);    // out = x * inv_rms
  out.multiply_i(gamma);        // out = x * inv_rms * gamma
}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen() + from * in_dim.width(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen() + from * out_dim.width(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();
      in_step.multiply(t, out_step);
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &incoming_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &outgoing_deriv =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &inv_rms = context.getTensor(wt_idx[RMSParams::inv_rms]);
  nntrainer::Tensor &temp = context.getTensor(wt_idx[RMSParams::temp_full]);

  // gamma_dy = gamma * incoming_derivative (element-wise)
  incoming_deriv.multiply(gamma, temp);

  // c = mean(gamma_dy * x) along width axis → shape (..., 1)
  // Reuse inv_rms shape for storing c: we need a temp scalar-per-row tensor
  // But inv_rms is needed, so use outgoing_deriv as temp for c computation
  // c_full = gamma_dy * x
  // c = sum(c_full) / width
  // dx = inv_rms * (gamma_dy - x * c * inv_rms^2)

  // outgoing_deriv = gamma_dy * x (element-wise)
  temp.multiply(input, outgoing_deriv);

  // sum along width → need a reduced tensor
  nntrainer::Tensor mean_val = outgoing_deriv.average(3);

  // inv_rms^2
  nntrainer::Tensor inv_rms_sq = inv_rms.multiply(inv_rms);

  // x * mean(gamma_dy * x) * inv_rms^2
  // outgoing_deriv = x * mean_val * inv_rms_sq
  input.multiply(mean_val, outgoing_deriv);
  outgoing_deriv.multiply_i(inv_rms_sq);

  // dx = inv_rms * (gamma_dy - x * mean(gamma_dy * x) * inv_rms^2)
  temp.subtract(outgoing_deriv, outgoing_deriv);
  outgoing_deriv.multiply_i(inv_rms);
}

void RMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);

  dgamma.setZero();

  ml::train::TensorDim in_dim = in.getDim();
  unsigned int b_size = in_dim.batch();
  unsigned int width = in_dim.width();

  ml::train::TensorDim in_step_dim = in_dim;
  in_step_dim.batch(1);

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor dy_step =
      dy.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const float *in_data = in_step.getData<float>();
      const float *dy_data = dy_step.getData<float>();
      float *dgamma_data = dgamma.getData<float>();
      
      unsigned int height = in_step_dim.height();
      
      for(unsigned int h = 0; h < height; ++h) {
          float sum_sq = 0.0f;
          for(unsigned int w = 0; w < width; ++w) {
              float val = in_data[h * width + w];
              sum_sq += val * val;
          }
          float mean_sq = sum_sq / width;
          float inv_sqrt_v = 1.0f / std::sqrt(mean_sq + epsilon);

          for(unsigned int w = 0; w < width; ++w) {
              dgamma_data[w] += dy_data[h * width + w] * in_data[h * width + w] * inv_sqrt_v;
          }
      }
    } else {
      throw std::invalid_argument("Error: not yet implemented for this data type");
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
