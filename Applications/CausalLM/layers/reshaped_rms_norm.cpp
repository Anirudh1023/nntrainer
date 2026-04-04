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
#include <reshaped_rms_norm.h>

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
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);

  unsigned int reshaped_height = dim[0].height() * (dim[0].width() / feature_size);

  nntrainer::TensorDim inv_rms_dim(
    dim[0].batch(), dim[0].channel(), reshaped_height, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::inv_rms] = context.requestTensor(
    inv_rms_dim, "inv_rms", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  nntrainer::TensorDim temp_dim = dim[0];
  temp_dim.height(reshaped_height);
  temp_dim.width(feature_size);
  wt_idx[RMSParams::temp_full] = context.requestTensor(
    temp_dim, "temp_full", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::CALC_DERIV_LIFESPAN);
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &inv_rms = context.getTensor(wt_idx[RMSParams::inv_rms]);

  ml::train::TensorDim original_dim = in.getDim();
  ml::train::TensorDim reshaped_dim = original_dim;
  reshaped_dim.width(feature_size);
  reshaped_dim.height(original_dim.height() * (original_dim.width() / feature_size));

  in.reshape(reshaped_dim);
  out.reshape(reshaped_dim);

  in.multiply(in, out);         // out = x^2 (temp use)
  out.average(3, inv_rms);      // inv_rms = mean(x^2)
  inv_rms.add_i(epsilon);       // inv_rms = mean(x^2) + eps
  inv_rms.inv_sqrt_i();         // inv_rms = 1/sqrt(mean(x^2) + eps)

  in.multiply(inv_rms, out);    // out = x * inv_rms
  out.multiply_i(gamma);        // out = x * inv_rms * gamma

  in.reshape(original_dim);
  out.reshape(original_dim);
}

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
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen() + from * in_dim.width(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen() + from * out_dim.width(), true);

    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
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

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

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
    nntrainer::Tensor dx_step =
      dx.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const float *in_data = in_step.getData<float>();
      const float *dy_data = dy_step.getData<float>();
      float *dx_data = dx_step.getData<float>();
      const float *gamma_data = gamma.getData<float>();

      unsigned int height = in_step_dim.height() * (width / feature_size);

      for(unsigned int h = 0; h < height; ++h) {
          float sum_sq = 0.0f;
          for(unsigned int w = 0; w < feature_size; ++w) {
              float val = in_data[h * feature_size + w];
              sum_sq += val * val;
          }
          float mean_sq = sum_sq / feature_size;
          float var_eps = mean_sq + epsilon;
          float inv_v = 1.0f / var_eps;
          float inv_sqrt_v = 1.0f / std::sqrt(var_eps);

          float sum_term = 0.0f;
          for(unsigned int w = 0; w < feature_size; ++w) {
              sum_term += (dy_data[h * feature_size + w] * gamma_data[w]) * in_data[h * feature_size + w];
          }

          for(unsigned int w = 0; w < feature_size; ++w) {
              float dy_g = dy_data[h * feature_size + w] * gamma_data[w];
              float term2 = (in_data[h * feature_size + w] * inv_v * sum_term) / feature_size;
              dx_data[h * feature_size + w] = (dy_g - term2) * inv_sqrt_v;
          }
      }
    } else {
      throw std::invalid_argument("Error: not yet implemented for this data type");
    }
  }
}

void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
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
      
      unsigned int height = in_step_dim.height() * (width / feature_size);
      
      for(unsigned int h = 0; h < height; ++h) {
          float sum_sq = 0.0f;
          for(unsigned int w = 0; w < feature_size; ++w) {
              float val = in_data[h * feature_size + w];
              sum_sq += val * val;
          }
          float mean_sq = sum_sq / feature_size;
          float inv_sqrt_v = 1.0f / std::sqrt(mean_sq + epsilon);

          for(unsigned int w = 0; w < feature_size; ++w) {
              dgamma_data[w] += dy_data[h * feature_size + w] * in_data[h * feature_size + w] * inv_sqrt_v;
          }
      }
    } else {
      throw std::invalid_argument("Error: not yet implemented for this data type");
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
