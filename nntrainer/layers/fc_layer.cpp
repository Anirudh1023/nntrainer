/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

#include <common_properties.h>
#include <fc_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

// Static registry: layer_name → QAT stats, updated every forward pass.
std::mutex FullyConnectedLayer::s_registry_mutex;
std::unordered_map<std::string, FullyConnectedLayer::LoRAQATStats>
  FullyConnectedLayer::s_qat_registry;

FullyConnectedLayer::FullyConnectedLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  q_min(-32.0f),   // Q6_K: 64 levels in [-32, 31], NOT INT8 [-128, 127]
  q_max(31.0f),
  momentum(0.1f),
  fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha(), props::LoraQAT()),
  quantizer(nullptr),
  qat_initialized_(false) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

FullyConnectedLayer::~FullyConnectedLayer() {
  if (qat_initialized_)
    printQATStats();
}

// Fake-quantize x to the [q_min_val, q_max_val] grid using EMA running stats.
// Matches Pranjal's qat_fc_layer.cpp design: parameterized range + momentum member.
//
// Training:  updates EMA with current-batch stats, quantizes using exact batch stats
//            (avoids clipping during early training when weights are large).
// Inference: quantizes using EMA stats (force-feed calibrated scale for nntr_quantize).
Tensor FullyConnectedLayer::fakeQuantize(const Tensor &x, Tensor &rmin,
                                          Tensor &rmax, float q_min_val,
                                          float q_max_val, bool training) {
  float cur_min = x.minValue();
  float cur_max = x.maxValue();

  if (training) {
    float rm = rmin.getValue<float>(0);
    float rx = rmax.getValue<float>(0);
    if (std::isinf(rm)) {
      rmin.setValue(cur_min);
      rmax.setValue(cur_max);
    } else {
      rmin.setValue((1.0f - momentum) * rm + momentum * cur_min);
      rmax.setValue((1.0f - momentum) * rx + momentum * cur_max);
    }
  } else {
    cur_min = rmin.getValue<float>(0);
    cur_max = rmax.getValue<float>(0);
  }

  float range = cur_max - cur_min;
  if (range < 1e-8f)
    range = 1e-8f;

  float scale      = range / (q_max_val - q_min_val);
  float zero_point = q_min_val - std::round(cur_min / scale);
  zero_point       = std::max(q_min_val, std::min(q_max_val, zero_point));

  Tensor x_fq = x.clone();
  std::function<float(float)> quantize_fn =
    [scale, zero_point, q_min_val, q_max_val](float v) -> float {
    float q = std::round(v / scale + zero_point);
    q        = std::max(q_min_val, std::min(q_max_val, q));
    return (q - zero_point) * scale;
  };
  x_fq.apply<float>(quantize_fn, x_fq);
  return x_fq;
}

void FullyConnectedLayer::printQATStats() const {
  const auto &lora_rank_prop = std::get<props::LoraRank>(fc_props);
  if (lora_rank_prop.empty())
    return;

  // Only print first 7 layers (one transformer block) to avoid flooding
  static int printed = 0;
  if (printed == 0)
    std::cerr << "\n[QAT] Final calibration stats (first transformer block):\n";
  if (printed++ >= 7)
    return;

  float a_min   = lora_a_rmin.getValue<float>(0);
  float a_max   = lora_a_rmax.getValue<float>(0);
  float a_scale = std::max(a_max - a_min, 1e-8f) / (q_max - q_min);

  float b_min   = lora_b_rmin.getValue<float>(0);
  float b_max   = lora_b_rmax.getValue<float>(0);
  float b_scale = std::max(b_max - b_min, 1e-8f) / (q_max - q_min);

  std::cerr << "  layer" << printed << ":"
            << " loraA scale=" << a_scale
            << " [" << a_min << ", " << a_max << "]"
            << " | loraB scale=" << b_scale
            << " [" << b_min << ", " << b_max << "]\n";
  std::cerr << std::flush;
}

FullyConnectedLayer::LoRAQATStats FullyConnectedLayer::getLoRAQATStats() const {
  LoRAQATStats s;
  if (!qat_initialized_)
    return s;
  s.a_min   = lora_a_rmin.getValue<float>(0);
  s.a_max   = lora_a_rmax.getValue<float>(0);
  s.a_scale = std::max(s.a_max - s.a_min, 1e-8f) / (q_max - q_min);
  s.b_min   = lora_b_rmin.getValue<float>(0);
  s.b_max   = lora_b_rmax.getValue<float>(0);
  s.b_scale = std::max(s.b_max - s.b_min, 1e-8f) / (q_max - q_min);
  s.valid   = true;
  return s;
}

FullyConnectedLayer::LoRAQATStats
FullyConnectedLayer::getRegisteredStats(const std::string &layer_name) {
  std::lock_guard<std::mutex> lock(s_registry_mutex);
  auto it = s_qat_registry.find(layer_name);
  if (it != s_qat_registry.end())
    return it->second;
  return {};
}

void FullyConnectedLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note bias is directly added to activation
  /// since we have no dequantizer for add operation,
  /// we have to set its data type as same as activation.
  /// This should be updated when the dequantizer is supported.
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  // Base weight is trainable only when LoRA is not active for this layer.
  // When lora_rank > 0, only loraA/loraB update; W is frozen.
  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", (lora_rank == 0));

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", (lora_rank == 0));
  }

  /** create weights for LoRA */
  if (lora_rank) {

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    // LoRA adapters are always FP32 regardless of the base weight dtype
    // (Q4_0/Q4_K etc. would reject rank=8 as width since 8 % 32 != 0)
    TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      TensorDim::TensorType(context.getFormat(), TensorDim::DataType::FP32),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      TensorDim::TensorType(context.getFormat(), TensorDim::DataType::FP32),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    // A=zeros, B=random: keeps zero LoRA contribution at init (0 @ random = 0)
    // but b_fq is non-zero from batch 1, so A gets gradient via chain-rule STE
    // immediately. Matches Pranjal's QAT design (qat_fc_layer.cpp).
    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);

    // Initialize QAT EMA running stats (scalar tensors, live in layer object)
    const bool lora_qat = !std::get<props::LoraQAT>(fc_props).empty() &&
                           std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat) {
      lora_a_rmin = Tensor({1});
      lora_a_rmax = Tensor({1});
      lora_b_rmin = Tensor({1});
      lora_b_rmax = Tensor({1});
      lora_a_rmin.setValue(std::numeric_limits<float>::infinity());
      lora_a_rmax.setValue(-std::numeric_limits<float>::infinity());
      lora_b_rmin.setValue(std::numeric_limits<float>::infinity());
      lora_b_rmax.setValue(-std::numeric_limits<float>::infinity());
      qat_initialized_ = true;
      layer_name_ = context.getName();
      static int qat_layer_count = 0;
      if (++qat_layer_count == 1)
        std::cerr << "[QAT] LoRA QAT active: q_range=[" << q_min << ", "
                  << q_max << "] (64 levels, Q6_K). "
                  << "Final EMA stats printed at exit.\n";
    }
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer =
      Quantization::createQuantizer(nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }
}

void FullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                   unsigned int batch) {
  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    Tensor weight_ = quantizer->dequantize(weight, input_.getDataType());
    input_.dot(weight_, hidden_, false, false);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    const bool lora_qat = !std::get<props::LoraQAT>(fc_props).empty() &&
                           std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat) {
      if (training) {
        // Training: update EMA stats and fake-quantize using current-batch range
        a_fq = fakeQuantize(loraA, lora_a_rmin, lora_a_rmax, q_min, q_max, true);
        b_fq = fakeQuantize(loraB, lora_b_rmin, lora_b_rmax, q_min, q_max, true);
        // Push current EMA stats into the global registry so transformer.cpp
        // can read them without a dynamic_cast across the .so boundary.
        {
          LoRAQATStats s = getLoRAQATStats();
          std::lock_guard<std::mutex> lk(s_registry_mutex);
          s_qat_registry[layer_name_] = s;
        }
      } else {
        // Inference/validation: use EMA-calibrated stats for fake-quantize.
        // Do NOT write back to loraA/loraB — that would corrupt Adam's momentum
        // state (force-feed at validation time fights the optimizer every epoch).
        // Weight snapping for nntr_quantize export is handled at save time only.
        a_fq = fakeQuantize(loraA, lora_a_rmin, lora_a_rmax, q_min, q_max, false);
        b_fq = fakeQuantize(loraB, lora_b_rmin, lora_b_rmax, q_min, q_max, false);
      }
      input_.dot(a_fq, hidden_tmp_lora, false, false);
      hidden_tmp_lora.dot(b_fq, hidden_out_lora, false, false);
    } else {
      input_.dot(loraA, hidden_tmp_lora, false, false);
      hidden_tmp_lora.dot(loraB, hidden_out_lora, false, false);
    }
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void FullyConnectedLayer::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor loraA, loraB, hidden_tmp_lora, hidden_out_lora;

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);
  }

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * hidden_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (!std::get<props::LoraRank>(fc_props).empty()) {
      nntrainer::TensorDim hidden_tmp_lora_step_dim = hidden_tmp_lora.getDim();
      hidden_tmp_lora_step_dim.batch(1);
      if (hidden_tmp_lora_step_dim.height() > 1)
        hidden_tmp_lora_step_dim.height(to - from);

      nntrainer::TensorDim hidden_out_lora_step_dim = hidden_out_lora.getDim();
      hidden_out_lora_step_dim.batch(1);
      if (hidden_out_lora_step_dim.height() > 1)
        hidden_out_lora_step_dim.height(to - from);

      nntrainer::Tensor hidden_tmp_lora_step =
        hidden_tmp_lora.getSharedDataTensor(
          hidden_tmp_lora_step_dim,
          b * hidden_tmp_lora.height() * hidden_tmp_lora.width(), true);
      nntrainer::Tensor hidden_out_lora_step =
        hidden_out_lora.getSharedDataTensor(
          hidden_out_lora_step_dim,
          b * hidden_out_lora.height() * hidden_out_lora.width(), true);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // MODE 2 (LoRA QAT): effective weight = W_frozen + a_fq · b_fq · scaling
    // dL/dx = dL/dy * [W + a_fq · b_fq · scaling]^T
    // Using a_fq/b_fq (from forward) matches Pranjal's qat_fc_layer reference.
    // Base is frozen in LoRA training so this gradient feeds no weight update,
    // but using a_fq/b_fq is theoretically correct for the forward computation.
    Tensor w_fp32;
    using DT = TensorDim::DataType;
    if (quantizer != nullptr) {
      Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
      w_fp32 = quantizer->dequantize(weight, lora_A.getDataType());
    } else if (weight.getDataType() == DT::Q4_0) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
      w_fp32 = dq->dequantize(weight, DT::FP32);
    } else if (weight.getDataType() == DT::Q6_K) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q6_K);
      w_fp32 = dq->dequantize(weight, DT::FP32);
    } else {
      w_fp32 = weight;
    }

    Tensor lora_contrib;
    if (qat_initialized_) {
      // chain-rule STE: dL/dx uses the same a_fq/b_fq that the forward used
      lora_contrib = a_fq.dot(b_fq).multiply(lora_scaling);
    } else {
      Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
      Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
      lora_contrib = lora_A.dot(lora_B).multiply(lora_scaling);
    }
    ret_.dot_deriv_wrt_1(w_fp32.add(lora_contrib), derivative_, false, false);
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {

  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

      if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
        derivative_.sum({0, 1, 2}, djdb);
      } else {
        /// @todo optimize below by adding beta to Tensor::sum
        Tensor t = derivative_.sum({0, 1, 2});
        djdb.add_i(t);
      }
    }

    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    // LoRA calcGradient with chain-rule STE.
    // QAT path: backward uses b_fq (from forward) for dL/dA, matching
    // the actual computation in forwarding. B=LECUN_NORMAL init ensures
    // b_fq is non-zero from batch 1, so A gets gradient immediately.
    // Non-QAT path: uses raw loraB as before.

    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));

    if (qat_initialized_) {
      // chain-rule STE: dL/d(loraTmp) = dL/dy_lora · b_fq^T
      djdtmp.dot_deriv_wrt_1(
        b_fq, lora_derivative_, false, false,
        !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    } else {
      Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
      djdtmp.dot_deriv_wrt_1(
        loraB, lora_derivative_, false, false,
        !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    }

    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} /* namespace nntrainer */
