// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <functional>
#include <layer_impl.h>
#include <limits>
#include <mutex>
#include <string>
#include <tensor.h>
#include <unordered_map>

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   * Prints final QAT calibration stats when lora_qat was active.
   */
  ~FullyConnectedLayer();

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer(FullyConnectedLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayer &operator=(FullyConnectedLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  static constexpr const char *type = "fully_connected";

  struct LoRAQATStats {
    float a_min = 0, a_max = 0, a_scale = 0;
    float b_min = 0, b_max = 0, b_scale = 0;
    bool valid = false;
  };
  LoRAQATStats getLoRAQATStats() const;

  /** Look up EMA stats for a layer by name (set during finalize). Thread-safe. */
  static LoRAQATStats getRegisteredStats(const std::string &layer_name);

private:
  static std::mutex s_registry_mutex;
  static std::unordered_map<std::string, LoRAQATStats> s_qat_registry;

  float lora_scaling;
  float q_min;    /**< Q6_K lower bound: -32 (64 levels, 6-bit) */
  float q_max;    /**< Q6_K upper bound:  31 */
  float momentum; /**< EMA momentum for running min/max stats */
  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha, props::LoraQAT>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_alpha - alpha for LoRA scaling
                                                lora_qat - enable Q6_K fake-quant on LoRA adapters */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
  std::unique_ptr<nntrainer::Quantizer> quantizer;

  bool qat_initialized_;    // true once finalize() sets up QAT EMA tensors
  std::string layer_name_;  // name captured in finalize(), used to key s_qat_registry

  // QAT: EMA running stats for loraA and loraB (scalar tensors, persist across batches)
  Tensor lora_a_rmin, lora_a_rmax;
  Tensor lora_b_rmin, lora_b_rmax;
  // QAT: cached fake-quantized adapters from last forward (used in STE)
  Tensor a_fq, b_fq;

  /**
   * @brief Fake-quantize x to Q6_K precision with EMA stats.
   *        q_min_val/q_max_val control the quantization grid (use q_min/q_max members).
   *        Training: updates EMA, quantizes with current-batch stats.
   *        Inference: quantizes with EMA stats and snaps weights in-place (force-feed).
   *        Backward is STE: gradient passes through unchanged.
   */
  Tensor fakeQuantize(const Tensor &x, Tensor &rmin, Tensor &rmax,
                      float q_min_val, float q_max_val, bool training);

  /**
   * @brief Print QAT calibration stats (EMA min/max and derived scale) for debugging.
   */
  void printQATStats() const;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
