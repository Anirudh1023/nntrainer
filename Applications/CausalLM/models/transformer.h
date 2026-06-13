// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.h
 * @date   31 Dec 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This transformer.h constructs a class for Transformer model which can
 * be a parent of CausalLM and Encoder models with transformer structure.
 * @note   This transformer assumes the following structure :
 *
 *           [Input]
 *              |
 *         [Embedding]
 *              |
 *        [Decoder Block] (repeated N times)
 *              |
 *          [RMSNorm]
 *
 */
#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <functional>
#include <layer.h>
#include <map>
#include <model.h>
#include <random>

#include <limits.h>

#include "json.hpp"
#include "performance_metrics.h"
#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace causallm {

/*** ALIAS ****/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

/**
 * @brief Transformer Class
 */
WIN_EXPORT class Transformer {

public:
  /**
   * @brief Construct a new Transformer object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation (generation.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   * @param model_type Type of the model (default: ModelType::MODEL)
   */
  Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
              ModelType model_type = ModelType::MODEL);

  /**
   * @brief Destroy the Transformer object
   */
  virtual ~Transformer() {}

  /**
   * @brief Initialize and Construct the Transformer model (inference mode)
   */
  virtual void initialize();

  /**
   * @brief Initialize model for LoRA fine-tuning (training mode).
   *        Adds cross_softmax loss, Adam optimizer, compiles in TRAIN mode.
   * @param lr   Learning rate for Adam optimizer
   * @param epochs Number of training epochs
   */
  virtual void initializeForTraining(float lr, unsigned int epochs);

  /**
   * @brief Load the model weights from a file
   */
  virtual void load_weight(const std::string &weight_path);

  /**
   * @brief Save the weight to a file
   */
  virtual void save_weight(const std::string &weight_path);

  /**
   * @brief Save only LoRA adapter weights (loraA/loraB) to a file.
   *        For use after LoRA training.
   */
  virtual void save_weight_lora(const std::string &weight_path);

  /**
   * @brief Load base weights, then overlay LoRA adapter weights on top.
   */
  virtual void load_weight_lora(const std::string &base_path,
                                const std::string &lora_path);

  /**
   * @brief Set a dataset on the underlying nntrainer model.
   */
  virtual void
  setDataset(const ml::train::DatasetModeType &mode,
             std::shared_ptr<ml::train::Dataset> dataset);

  /**
   * @brief Run training on the model (wraps model->train()).
   */
  virtual void train();

  /**
   * @brief Run training with an epoch-end callback and optional early-stop
   *        predicate.
   * @param epoch_cb   Called at the end of each epoch (after stats are updated).
   * @param epoch_data Passed as-is to epoch_cb.
   * @param stop_cb    Returns true to stop training early. nullptr = never stop.
   * @param stop_data  Passed as-is to stop_cb.
   */
  virtual void train(std::function<void(void *)> epoch_cb, void *epoch_data,
                     std::function<bool(void *)> stop_cb = nullptr,
                     void *stop_data = nullptr);

  /**
   * @brief Return training stats from the last completed epoch.
   */
  virtual ml::train::RunStats getTrainingStats();

  /**
   * @brief Return validation stats from the last completed epoch.
   */
  virtual ml::train::RunStats getValidStats();

  /**
   * @brief Print model summary to a stream.
   */
  virtual void summarize(std::ostream &out, unsigned int type);

  /**
   * @brief Export weight names and norms to a text file for debugging.
   */
  virtual void exportWeightsToFile(const std::string &path);

  /**
   * @brief Save the weight to a file with type conversion
   * @param weight_path Path to save the weight file
   * @param dtype Global target data type for all layers (NONE = keep original)
   * @param layer_dtype_map Per-layer data type overrides (layer_name -> dtype)
   */
  virtual void
  save_weight(const std::string &weight_path,
              ml::train::TensorDim::DataType dtype,
              const std::map<std::string, ml::train::TensorDim::DataType>
                &layer_dtype_map = {});

  /**
   * @brief run the Transformer model
   */
  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = "", const WSTR tail_prompt = "",
                   bool log_output = true);

  /**
   * @brief Get PerformanceMetrics
   */
  PerformanceMetrics getPerformanceMetrics() const {
    return performance_metrics;
  }

protected:
  /**
   * @brief Setup the parameters for the Transformer model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Construct Model
   */
  virtual void constructModel();

  /**
   * @brief create Attention Layer
   */
  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  /**
   * @brief create Attention Layer
   */
  virtual std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name);

  /**
   * @brief create Feed Forward Layer
   */
  virtual std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                             int hidden_dim,
                                             std::string input_name);

  /**
   * @brief register CustomLayers
   */
  virtual void registerCustomLayers();

  /**
   * @brief Returns true if module_type (e.g. "q_proj") is in LORA_TARGET and
   * LORA_RANK > 0.
   */
  bool hasLoRA(const std::string &module_type) const;

  /**
   * @brief Append lora_rank (and lora_alpha if set) to a layer property list.
   */
  void appendLoRAProps(std::vector<std::string> &props) const;

  /**
   * @brief register Outputs
   */
  bool is_initialized = false; /**< Flag to check if the model is initialized */
  ModelHandle model;

  /** tokenizer */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;

  unsigned int NUM_VOCAB;
  int DIM;
  int HEAD_DIM;
  int INTERMEDIATE_SIZE;
  int NUM_LAYERS;
  bool USE_VOCAB_SELECTION;
  bool TIE_WORD_EMBEDDINGS;
  unsigned int MAX_SEQ_LEN;
  int NUM_HEADS;
  int NUM_KEY_VALUE_HEADS;
  int NUM_TO_GENERATE;
  std::string MODEL_TENSOR_TYPE;
  std::string EMBEDDING_DTYPE; /** embedding dtype */
  std::string FC_LAYER_DTYPE;  /** custom_fc_lora */

  unsigned int SLIDING_WINDOW = UINT_MAX;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
  unsigned int ROPE_THETA = 10000; /**< RoPE theta value */
  float NORM_EPS = 1e-5;           /**< RMSNorm epsilon value */
  float EMBEDDING_SCALE = 1.0f;
  int GQA_SIZE;

  unsigned int BATCH_SIZE;              /**< Batch size for the model */
  unsigned int INIT_SEQ_LEN;            /**< Initial sequence length */
  unsigned int MAX_POSITION_EMBEDDINGS; /**< max_position embeddings */
  bool MEMORY_SWAP;                     /**< memory swap option */
  unsigned int FSU_LOOKAHEAD;
  float ATTN_LOGIT_SOFTCAPPING = 0.0f; /**< attention logit softcapping */
  bool IS_CAUSAL = true;

  unsigned int LORA_RANK = 0;  /**< LoRA rank (0 = disabled) */
  unsigned int LORA_ALPHA = 0; /**< LoRA alpha (0 = use scaling=1) */
  std::vector<std::string> LORA_TARGET; /**< module names to apply LoRA to,
                                            e.g. {"q_proj","v_proj"} */

  // Performance metrics
  PerformanceMetrics performance_metrics;
};
/**
 * Loads JSON data from a file with detailed error handling
 * @param file_path Path to JSON file
 * @return JSON object
 * @throws std::runtime_error on file open or parse failure
 */
std::string LoadBytesFromFile(const std::string &path);

inline json LoadJsonFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path +
                             " | Reason: " + std::strerror(errno));
  }

  try {
    json data;
    file >> data;
    return data;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error in " + file_path +
                             " | Details: " + e.what());
  }
}
} // namespace causallm

#endif
