// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <unordered_set>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  // Initialize the model with the provided configurations
  // This is where you would set up the model layers, parameters, etc.
  setupParameters(cfg, generation_cfg, nntr_cfg);

  // prep tokenizer
  tokenizer = tokenizers::Tokenizer::FromBlobJSON(
    LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
};

void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention")) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  }

  NUM_VOCAB = cfg["vocab_size"];
  DIM = cfg["hidden_size"];
  INTERMEDIATE_SIZE = cfg["intermediate_size"];
  NUM_LAYERS = cfg["num_hidden_layers"];
  NUM_HEADS = cfg["num_attention_heads"];
  HEAD_DIM = cfg.contains("head_dim")
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS; // default value is hidden_size / num_heads
  NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
  ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
  NORM_EPS = cfg["rms_norm_eps"];
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  LORA_RANK = nntr_cfg.contains("lora_rank")
                ? nntr_cfg["lora_rank"].get<unsigned int>()
                : 0;
  LORA_ALPHA = nntr_cfg.contains("lora_alpha")
                 ? nntr_cfg["lora_alpha"].get<unsigned int>()
                 : 0;
  LORA_TARGET =
    nntr_cfg.contains("lora_target")
      ? nntr_cfg["lora_target"].get<std::vector<std::string>>()
      : std::vector<std::string>{};

  return;
};

bool Transformer::hasLoRA(const std::string &module_type) const {
  if (LORA_RANK == 0 || LORA_TARGET.empty())
    return false;
  return std::find(LORA_TARGET.begin(), LORA_TARGET.end(), module_type) !=
         LORA_TARGET.end();
}

void Transformer::appendLoRAProps(std::vector<std::string> &props) const {
  props.push_back(withKey("lora_rank", LORA_RANK));
  if (LORA_ALPHA > 0)
    props.push_back(withKey("lora_alpha", LORA_ALPHA));
}

void Transformer::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

  // construct causalLM model
  constructModel();

  // setup model property
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

void Transformer::constructModel() {

  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer (frozen in LoRA mode)
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  {
    std::vector<std::string> emb_params = {
      "name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
      "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
      "scale=" + std::to_string(EMBEDDING_SCALE)};
    if (LORA_RANK > 0)
      emb_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer(embedding_type, emb_params));
  }

  // create transformer layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoderBlock(0, "embedding0");
    else
      transformer = createTransformerDecoderBlock(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // output rms_norm (frozen in LoRA mode)
  {
    std::vector<std::string> norm_params = {
      withKey("name", "output_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("input_layers",
              "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
      withKey("packed", "false")};
    if (LORA_RANK > 0)
      norm_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("rms_norm", norm_params));
  }

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
};

void Transformer::initializeForTraining(float lr, unsigned int epochs) {
  registerCustomLayers();
  constructModel();

  try {
    model->addLayer(
      ml::train::createLayer("cross_softmax", {"name=loss"}));
  } catch (const std::exception &e) {
    std::cerr << "[initializeForTraining] loss layer: " << e.what() << std::endl;
  }

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", epochs),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  model->setProperty(model_props);

  auto optimizer =
    ml::train::createOptimizer("adam", {"learning_rate=" + std::to_string(lr)});
  if (model->setOptimizer(std::move(optimizer)))
    throw std::invalid_argument("Failed to set optimizer.");

  int compile_ret = model->compile(ml::train::ExecutionMode::TRAIN);
  if (compile_ret) {
    std::cerr << "[initializeForTraining] compile() returned " << compile_ret << std::endl;
    throw std::invalid_argument("Model compilation for training failed.");
  }

  int init_ret = model->initialize(ml::train::ExecutionMode::TRAIN);
  if (init_ret) {
    std::cerr << "[initializeForTraining] initialize() returned " << init_ret << std::endl;
    throw std::invalid_argument("Model initialization for training failed.");
  }

  is_initialized = true;
}

// Returns the ordered layer names matching NeuralNetwork::save() graph
// traversal. Used by load_weight, save_weight_lora, and load_weight_lora to
// iterate weights in a consistent, deterministic order.
static std::vector<std::string> buildOrderedLayerNames(int num_layers) {
  std::vector<std::string> names;
  names.push_back("embedding0");
  for (int i = 0; i < num_layers; ++i) {
    std::string p = "layer" + std::to_string(i);
    names.push_back(p + "_attention_norm");
    names.push_back(p + "_wq");
    names.push_back(p + "_q_norm");
    names.push_back(p + "_wk");
    names.push_back(p + "_k_norm");
    names.push_back(p + "_wv");
    names.push_back(p + "_mha_core" + std::to_string(i));
    names.push_back(p + "_attention_out");
    names.push_back(p + "_ffn_norm");
    names.push_back(p + "_ffn_up");
    names.push_back(p + "_ffn_gate");
    names.push_back(p + "_ffn_down");
    names.push_back(p + "_swiglu");
    names.push_back(p + "_attention_add");
    names.push_back(p + "_ffn_add");
  }
  names.push_back("output_norm");
  names.push_back("output_of_causallm");
  return names;
}

void Transformer::load_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  // The pretrained BIN file was saved WITHOUT LoRA adapter slots.
  // model->load() assigns offsets positionally, so every loraA/loraB weight
  // in the LoRA model shifts subsequent base weights by ~32-64 KB, completely
  // scrambling the loaded pretrained weights.
  //
  // Fix: read the file manually, advancing the file pointer only for base
  // weights (not loraA/loraB), so each base weight reads from its correct
  // position in the pretrained file.
  std::ifstream f(weight_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open model weights: " + weight_path);

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);

  std::unordered_set<float *> visited;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi])
        continue;
      // Deduplicate shared tensors (e.g. TieWordEmbedding "Embedding").
      if (!visited.insert(wdata[wi]).second)
        continue;

      const std::string &wname = layer->getWeightName(wi);
      bool is_lora = (wname.find(":loraA") != std::string::npos ||
                      wname.find(":loraB") != std::string::npos);
      if (is_lora)
        continue; // Skip: not in pretrained file. Keeps initialized value.

      size_t bytes = static_cast<size_t>(wdims[wi].getDataLen()) * sizeof(float);
      f.read(reinterpret_cast<char *>(wdata[wi]), bytes);
      if (!f)
        throw std::runtime_error("load_weight: read failed at weight '" +
                                 wname + "' (offset " +
                                 std::to_string(f.tellg()) + ")");
    }
  }
  std::cout << "[load_weight] Loaded base weights from " << weight_path
            << " (LoRA adapters kept at initialized values)\n";
};

void Transformer::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

void Transformer::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType>
    &layer_dtype_map) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN, dtype,
                layer_dtype_map);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights with dtype: " +
                             std::string(e.what()));
  }
};

void Transformer::save_weight_lora(const std::string &weight_path) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before save_weight_lora().");

  std::ofstream f(weight_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open " + weight_path + " for writing.");

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);
  std::unordered_set<float *> visited;
  size_t total_bytes = 0;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0) continue;
    } catch (...) { continue; }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try { layer->getWeights(wdata, wdims); } catch (...) { continue; }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi]) continue;
      if (!visited.insert(wdata[wi]).second) continue;

      const std::string &wname = layer->getWeightName(wi);
      if (wname.find(":loraA") == std::string::npos &&
          wname.find(":loraB") == std::string::npos)
        continue;

      size_t bytes = static_cast<size_t>(wdims[wi].getDataLen()) * sizeof(float);
      f.write(reinterpret_cast<const char *>(wdata[wi]), bytes);
      total_bytes += bytes;
    }
  }

  std::cout << "[save_weight_lora] Saved LoRA adapters to " << weight_path
            << " (" << (total_bytes / 1024 / 1024) << " MB)\n";
}

void Transformer::load_weight_lora(const std::string &base_path,
                                   const std::string &lora_path) {
  load_weight(base_path);

  std::ifstream f(lora_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open LoRA adapters: " + lora_path);

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);
  std::unordered_set<float *> visited;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0) continue;
    } catch (...) { continue; }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try { layer->getWeights(wdata, wdims); } catch (...) { continue; }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi]) continue;
      if (!visited.insert(wdata[wi]).second) continue;

      const std::string &wname = layer->getWeightName(wi);
      if (wname.find(":loraA") == std::string::npos &&
          wname.find(":loraB") == std::string::npos)
        continue;

      size_t bytes = static_cast<size_t>(wdims[wi].getDataLen()) * sizeof(float);
      f.read(reinterpret_cast<char *>(wdata[wi]), bytes);
      if (!f)
        throw std::runtime_error("load_weight_lora: read failed at '" + wname + "'");
    }
  }

  std::cout << "[load_weight_lora] Loaded LoRA adapters from " << lora_path << "\n";
}

void Transformer::setDataset(const ml::train::DatasetModeType &mode,
                              std::shared_ptr<ml::train::Dataset> dataset) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before setDataset().");
  if (model->setDataset(mode, dataset))
    throw std::runtime_error("Failed to set dataset on model.");
}

void Transformer::train() {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before train().");
  if (model->train())
    throw std::runtime_error("model->train() returned error.");
}

void Transformer::summarize(std::ostream &out, unsigned int type) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before summarize().");
  model->summarize(out, static_cast<ml_train_summary_type_e>(type));
}

void Transformer::exportWeightsToFile(const std::string &path) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before exportWeightsToFile().");
  std::ofstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open " + path + " for weight export.");

  std::vector<std::string> layer_names;
  layer_names.push_back("embedding0");
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::string p = "layer" + std::to_string(i);
    layer_names.push_back(p + "_attention_norm");
    layer_names.push_back(p + "_wq");
    layer_names.push_back(p + "_q_norm");   // Qwen3 QK norm
    layer_names.push_back(p + "_wk");
    layer_names.push_back(p + "_k_norm");   // Qwen3 QK norm
    layer_names.push_back(p + "_wv");
    layer_names.push_back(p + "_attention_out");
    layer_names.push_back(p + "_ffn_norm");
    layer_names.push_back(p + "_ffn_up");
    layer_names.push_back(p + "_ffn_gate");
    layer_names.push_back(p + "_ffn_down");
  }
  layer_names.push_back("output_norm");
  layer_names.push_back("output_of_causallm");

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      try {
        const std::string &wname = layer->getWeightName(wi);
        unsigned int n = wdims[wi].getDataLen();
        double norm = 0.0;
        if (wdata[wi]) {
          for (unsigned int k = 0; k < n; ++k)
            norm += static_cast<double>(wdata[wi][k]) * wdata[wi][k];
          norm = std::sqrt(norm);
        }
        f << lname << "/" << wname << ": " << std::fixed
          << std::setprecision(6) << norm << "\n";
      } catch (...) {
        continue;
      }
    }
  }
}

void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

std::vector<LayerHandle>
Transformer::createTransformerDecoderBlock(const int layer_id,
                                           std::string input_name) {

  std::vector<LayerHandle> layers;

  {
    std::vector<std::string> attn_norm_params = {
      withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
      withKey("input_layers", input_name),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("packed", "false")};
    if (LORA_RANK > 0)
      attn_norm_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("rms_norm", attn_norm_params));
  }

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");

  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  {
    std::vector<std::string> ffn_norm_params = {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
      withKey("input_layers",
              "layer" + std::to_string(layer_id) + "_decoder_add"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("packed", "false")};
    if (LORA_RANK > 0)
      ffn_norm_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("rms_norm", ffn_norm_params));
  }

  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle>
Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                             int head_dim, std::string query_name,
                             std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wq"))
    appendLoRAProps(q_params);
  else if (LORA_RANK > 0)
    q_params.push_back(withKey("trainable", "false"));
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wk"))
    appendLoRAProps(k_params);
  else if (LORA_RANK > 0)
    k_params.push_back(withKey("trainable", "false"));
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wv"))
    appendLoRAProps(v_params);
  else if (LORA_RANK > 0)
    v_params.push_back(withKey("trainable", "false"));
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                ? SLIDING_WINDOW
                                : UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  if (hasLoRA("wo"))
    appendLoRAProps(o_params);
  else if (LORA_RANK > 0)
    o_params.push_back(withKey("trainable", "false"));
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Transformer::createMlp(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name) {

  std::vector<LayerHandle> layers;

  {
    std::vector<std::string> up_params = {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
      withKey("unit", hidden_dim), withKey("disable_bias", "true"),
      withKey("input_layers", input_name),
      withKey("weight_initializer", "ones")};
    if (hasLoRA("ffn_up"))
      appendLoRAProps(up_params);
    else if (LORA_RANK > 0)
      up_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("fully_connected", up_params));
  }
  {
    std::vector<std::string> gate_params = {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
      withKey("unit", hidden_dim), withKey("disable_bias", "true"),
      withKey("input_layers", input_name),
      withKey("weight_initializer", "ones")};
    if (hasLoRA("ffn_gate"))
      appendLoRAProps(gate_params);
    else if (LORA_RANK > 0)
      gate_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("fully_connected", gate_params));
  }

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_gate," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_up")}));

  {
    std::vector<std::string> down_params = {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
      withKey("unit", dim), withKey("disable_bias", "true"),
      withKey("input_layers",
              "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
      withKey("weight_initializer", "ones")};
    if (hasLoRA("ffn_down"))
      appendLoRAProps(down_params);
    else if (LORA_RANK > 0)
      down_params.push_back(withKey("trainable", "false"));
    layers.push_back(createLayer("fully_connected", down_params));
  }

  return layers;
}

void Transformer::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);

  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
