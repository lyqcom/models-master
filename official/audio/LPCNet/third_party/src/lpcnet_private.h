#ifndef LPCNET_PRIVATE_H
#define LPCNET_PRIVATE_H

#include <stdio.h>
#include "common.h"
#include "freq.h"
#include "celt_lpc.h"
#include "lpcnet.h"

#define BITS_PER_CHAR 8

#define PITCH_MIN_PERIOD 32
#define PITCH_MAX_PERIOD 256

#define PITCH_FRAME_SIZE 320
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define MULTI 4
#define MULTI_MASK (MULTI-1)

#define FORBIDDEN_INTERP 7

#define FEATURES_DELAY (FEATURE_CONV1_DELAY + FEATURE_CONV2_DELAY)


struct LPCNetEncState{
  float analysis_mem[OVERLAP_SIZE];
  float mem_preemph;
  int pcount;
  float pitch_mem[LPC_ORDER];
  float pitch_filt;
  float xc[10][PITCH_MAX_PERIOD+1];
  float frame_weight[10];
  float exc_buf[PITCH_BUF_SIZE];
  float pitch_max_path[2][PITCH_MAX_PERIOD];
  float pitch_max_path_all;
  int best_i;
  float last_gain;
  int last_period;
  float lpc[LPC_ORDER];
  float vq_mem[NB_BANDS];
  float features[4][NB_TOTAL_FEATURES];
  float sig_mem[LPC_ORDER];
  int exc_mem;
};


extern float ceps_codebook1[];
extern float ceps_codebook2[];
extern float ceps_codebook3[];
extern float ceps_codebook_diff4[];

void preemphasis(float *y, float *mem, const float *x, float coef, int N);

void perform_double_interp(float features[4][NB_TOTAL_FEATURES], const float *mem, int best_id);

void process_superframe(LPCNetEncState *st, unsigned char *buf, FILE *ffeat, int encode, int quantize);

void compute_frame_features(LPCNetEncState *st, const float *in);

void decode_packet(float features[4][NB_TOTAL_FEATURES], float *vq_mem, const unsigned char buf[8]);

void process_single_frame(LPCNetEncState *st, FILE *ffeat);

void run_frame_network(LPCNetState *lpcnet, float *gru_a_condition, float *gru_b_condition, float *lpc, const float *features);
#endif
