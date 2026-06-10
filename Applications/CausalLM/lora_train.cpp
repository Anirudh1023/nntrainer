// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline implementation
 */

#include "lora_train.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size) :
  tokenizer_(tokenizer),
  seq_len_(seq_len),
  vocab_size_(vocab_size),
  current_idx_(0) {}

void TrainingDataGenerator::loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open training data file: " + path);

  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    auto ids = tokenizer_->Encode(line);
    samples_.push_back(ids);
    count++;
  }
  std::cout << "[TrainingData] Loaded " << path << " — " << count
            << " samples." << std::endl;
}

void TrainingDataGenerator::addTokenIds(const std::vector<int> &ids) {
  samples_.push_back(ids);
}

unsigned int TrainingDataGenerator::getNumSamples() const {
  return static_cast<unsigned int>(samples_.size());
}

void TrainingDataGenerator::reset() { current_idx_ = 0; }

void TrainingDataGenerator::limitSamples(unsigned int max_samples) {
  if (max_samples < samples_.size())
    samples_.resize(max_samples);
}

int TrainingDataGenerator::dataCb(float **input, float **label, bool *last,
                                  void *user_data) {
  auto *self = static_cast<TrainingDataGenerator *>(user_data);

  // Auto-reset at the start of each new epoch
  if (self->current_idx_ >= self->samples_.size())
    self->reset();

  const auto &ids = self->samples_[self->current_idx_];
  unsigned int available = static_cast<unsigned int>(ids.size());

  // Label = last token in the sequence (e.g. "Positive" / "Negative").
  // Input = everything before it, LEFT-padded to seq_len so the last real
  // token sits at position seq_len-1 (where lm_head reads from).
  // Right-padding would put the last real token at an early position followed
  // by many padding tokens, making the lm_head predict from a padding state.

  unsigned int label_token = 0;

  if (available >= 2) {
    label_token = static_cast<unsigned int>(ids[available - 1]);

    unsigned int input_len = available - 1;
    unsigned int start = (input_len > self->seq_len_) ? (input_len - self->seq_len_) : 0;
    unsigned int used = input_len - start;
    unsigned int pad  = self->seq_len_ - used;

    for (unsigned int j = 0; j < pad; ++j)
      input[0][j] = 0.0f;
    for (unsigned int j = 0; j < used; ++j)
      input[0][pad + j] = static_cast<float>(ids[start + j]);
  } else {
    for (unsigned int j = 0; j < self->seq_len_; ++j)
      input[0][j] = 0.0f;
  }

  for (unsigned int v = 0; v < self->vocab_size_; ++v)
    label[0][v] = 0.0f;
  if (label_token < self->vocab_size_)
    label[0][label_token] = 1.0f;

  std::cout << "[DataGen] sample " << self->current_idx_ << " / "
            << self->samples_.size() << "\r" << std::flush;

  self->current_idx_++;
  *last = (self->current_idx_ >= self->samples_.size());
  return 0;
}

} // namespace causallm
