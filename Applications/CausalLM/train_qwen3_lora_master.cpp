// SPDX-License-Identifier: Apache-2.0
/**
 * @file   train_qwen3_lora_master.cpp
 * @brief  Entry point for Qwen3-0.6B LoRA fine-tuning
 *
 * Usage:
 *   train_qwen3_lora_master <model_dir> <train_data.txt>
 *       [--lr <float>] [--epochs <int>]
 *       [--output <path>] [--lora_path <path>]
 *       [--max_samples <int>] [--skip_weights]
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <causal_lm.h>
#include <dataset.h>
#include <lora_train.h>
#include <model.h>
#include <transformer.h>

#include "json.hpp"
#include "qwen3_causallm.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt>"
                 " [--lr <float>] [--epochs <int>]"
                 " [--output <path>] [--lora_path <path>]"
                 " [--max_samples <int>] [--skip_weights]\n";
    return 1;
  }

  std::string model_dir       = argv[1];
  std::string train_data_path = argv[2];
  float lr                    = 1e-4f;
  unsigned int epochs         = 1;
  std::string output_path     = "lora_weights.bin";
  std::string lora_path;
  int max_samples   = -1;
  bool skip_weights = false;
  unsigned int patience = 5;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--lr" && i + 1 < argc)
      lr = std::atof(argv[++i]);
    else if (arg == "--epochs" && i + 1 < argc)
      epochs = static_cast<unsigned int>(std::atoi(argv[++i]));
    else if (arg == "--output" && i + 1 < argc)
      output_path = argv[++i];
    else if (arg == "--lora_path" && i + 1 < argc)
      lora_path = argv[++i];
    else if (arg == "--max_samples" && i + 1 < argc)
      max_samples = std::atoi(argv[++i]);
    else if (arg == "--skip_weights")
      skip_weights = true;
    else if (arg == "--patience" && i + 1 < argc)
      patience = static_cast<unsigned int>(std::atoi(argv[++i]));
  }

  try {
    std::string config_path    = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg      = causallm::LoadJsonFile(config_path);
    auto gen_cfg  = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    std::cout << "=== Qwen3 LoRA Training ===\n";
    std::cout << "Model dir : " << model_dir << "\n";
    std::cout << "Train data: " << train_data_path << "\n";
    std::cout << "LR=" << lr << "  epochs=" << epochs
              << "  patience=" << patience << "\n\n";

    // Inject LoRA config into nntr_cfg (override JSON in memory)
    if (!nntr_cfg.contains("lora_rank") || nntr_cfg["lora_rank"] == 0) {
      std::cout << "[LoRA] Injecting default LoRA config (rank=8, alpha=16).\n";
      nntr_cfg["lora_rank"]  = 8;
      nntr_cfg["lora_alpha"] = 16;
      nntr_cfg["lora_target"] =
        json::array({"wq", "wk", "wv", "wo", "ffn_up", "ffn_down", "ffn_gate"});
    }
    std::cout << "[LoRA] rank=" << nntr_cfg["lora_rank"]
              << "  alpha=" << nntr_cfg["lora_alpha"]
              << "  targets=" << nntr_cfg["lora_target"].dump() << "\n\n";

    auto model = std::make_unique<causallm::Qwen3CausalLM>(cfg, gen_cfg, nntr_cfg);
    model->initializeForTraining(lr, epochs);

    if (!skip_weights && nntr_cfg.contains("model_file_name")) {
      std::string weight_path =
        model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
      std::cout << "Loading weights: " << weight_path << "\n";
      if (!lora_path.empty()) {
        std::cout << "Loading LoRA overlay: " << lora_path << "\n";
        model->load_weight_lora(weight_path, lora_path);
      } else {
        model->load_weight(weight_path);
      }
    } else {
      std::cout << "Skipping weight load (random init).\n";
    }

    // Build tokenizer
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    if (nntr_cfg.contains("tokenizer_file"))
      tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    auto blob      = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);

    unsigned int seq_len   = nntr_cfg["init_seq_len"].get<unsigned int>();
    unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    causallm::TrainingDataGenerator data_gen(tokenizer.get(), seq_len, vocab_size);
    data_gen.loadTextFile(train_data_path);

    if (max_samples > 0 &&
        static_cast<unsigned int>(max_samples) < data_gen.getNumSamples()) {
      std::cout << "Limiting to " << max_samples << " samples.\n";
      data_gen.limitSamples(static_cast<unsigned int>(max_samples));
    }
    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: no training samples loaded.\n";
      return 1;
    }
    std::cout << "Training samples: " << data_gen.getNumSamples() << "\n\n";

    auto dataset = std::shared_ptr<ml::train::Dataset>(
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::TrainingDataGenerator::dataCb,
                               &data_gen));
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);
    // Same data used for accuracy tracking after each epoch
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset);

    // Epoch callback: cumulative loss + perplexity + early stopping.
    // Accuracy is always ~0% for next-token top-1 with vocab_size=151936;
    // perplexity (exp(loss)) is the meaningful language-model metric.
    struct CumStats {
      causallm::Qwen3CausalLM *mdl;
      unsigned int epoch_count  = 0;
      float cumulative_loss     = 0.0f;
      // early stopping
      unsigned int patience;
      unsigned int patience_left;
      float best_val_loss       = std::numeric_limits<float>::max();
      unsigned int best_epoch   = 0;
      bool stop_flag            = false;
      std::string output_path;
    };
    CumStats cum{model.get()};
    cum.patience      = patience;
    cum.patience_left = patience;
    cum.output_path   = output_path;

    auto epoch_cb = [](void *ud) {
      auto *c = static_cast<CumStats *>(ud);
      c->epoch_count++;
      auto ts = c->mdl->getTrainingStats();
      auto vs = c->mdl->getValidStats();
      c->cumulative_loss += ts.loss;
      float avg     = c->cumulative_loss / static_cast<float>(c->epoch_count);
      float ppl     = std::exp(ts.loss);
      float cum_ppl = std::exp(avg);
      std::cout << "  Cumulative | AvgLoss: " << avg
                << "  CumPPL: " << cum_ppl
                << "  EpochPPL: " << ppl << "\n";

      // Early stopping: track best validation loss
      if (vs.loss < c->best_val_loss) {
        c->best_val_loss   = vs.loss;
        c->best_epoch      = c->epoch_count;
        c->patience_left   = c->patience;
        c->mdl->save_weight_lora(c->output_path);
        std::cout << "  [Best] val_loss=" << vs.loss
                  << " at epoch " << c->epoch_count
                  << " -> checkpoint saved\n";
      } else {
        c->patience_left--;
        std::cout << "  [EarlyStopping] No improvement. patience="
                  << c->patience_left << "/" << c->patience << "\n";
        if (c->patience_left == 0)
          c->stop_flag = true;
      }
    };

    auto stop_cb = [](void *ud) -> bool {
      return static_cast<CumStats *>(ud)->stop_flag;
    };

    std::cout << "\n=== Starting training ===\n";
    auto t0 = std::chrono::steady_clock::now();
    model->train(epoch_cb, &cum, stop_cb, &cum);
    double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "\nTraining done in " << elapsed << " s.\n";
    std::cout << "Best checkpoint: epoch " << cum.best_epoch
              << "  val_loss=" << cum.best_val_loss
              << "  val_PPL=" << std::exp(cum.best_val_loss) << "\n";
    std::cout << "LoRA weights saved to: " << output_path << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
